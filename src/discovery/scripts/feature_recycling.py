import argparse
from dataclasses import dataclass
import os
import random
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb


@dataclass
class FeatureInfo:
    """Stores information about a feature including its utility and distribution parameters."""
    is_real: bool
    utility: float
    distribution_params: dict
    last_update: int
    creation_step: int

class DummyTask:
    def __init__(self, feature_dim: int, n_classes: int, task_type: str = 'classification'):
        """Initialize data generator.
        
        Args:
            feature_dim: Number of input features
            n_classes: Number of classes for classification
            task_type: Type of task ('classification' or 'regression')
        """
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.task_type = task_type
        
        # Set distributions for each feature
        self.distributions = [random.choice(['uniform', 'normal']) 
                            for _ in range(feature_dim)]

    def get_iterator(self, batch_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Get iterator that generates infinite batches of data."""
        while True:
            # Generate features
            features = []
            for dist in self.distributions:
                if dist == 'uniform':
                    feature = torch.rand(batch_size) * 2 - 1  # Uniform in [-1, 1]
                else:
                    feature = torch.randn(batch_size).clamp(-1, 1)  # Normal clamped to [-1, 1]
                features.append(feature)
            
            inputs = torch.stack(features, dim=1)
            
            # Generate targets
            if self.task_type == 'classification':
                targets = torch.randint(0, self.n_classes, (batch_size,))
            else:
                targets = torch.randn(batch_size)
            
            yield inputs, targets


class GEOFFTask:
    def __init__(
        self, 
        feature_dim: int = 20, 
        sign_flip_interval: int = 20,
        active_features: int = 5,
    ):
        """Initialize tracking task where target is sum of first k inputs with changing signs.
        
        Args:
            feature_dim: Number of input features (default 20)
            sign_flip_interval: How often to flip signs (in steps)
            active_features: Number of features that contribute to target (default 5)
        """
        self.feature_dim = feature_dim
        self.sign_flip_interval = sign_flip_interval
        self.active_features = active_features
        
        # Initialize signs randomly for active features (+1 or -1)
        self.signs = torch.randint(0, 2, (active_features,)) * 2 - 1
        
        self.steps_since_flip = 0
        self.task_type = 'regression'
    
    def _randomize_sign(self):
        """Randomly flip the sign of one random feature."""
        # Choose random feature to flip
        feature_to_flip = random.randrange(self.active_features)
        # Flip its sign
        self.signs[feature_to_flip] *= -1
    
    def get_iterator(self, batch_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Get iterator that generates infinite batches of data."""
        while True:
            # Generate all features from normal distribution
            inputs = torch.randn(batch_size, self.feature_dim)
            # Calculate target using only first k features with signs
            targets = (self.signs.unsqueeze(0) * inputs[:, :self.active_features]).sum(dim=1)
            
            # Update sign flip counter and flip if needed
            self.steps_since_flip += 1
            if self.sign_flip_interval > 0 and self.steps_since_flip >= self.sign_flip_interval:
                self._randomize_sign()
                self.steps_since_flip = 0
            
            yield inputs, targets


class RecyclingMLP(nn.Module):
    def __init__(
        self, 
        input_size: int,
        n_classes: int,
        n_layers: int,
        hidden_dim: int,
        weight_init_method: str,
        device: str = 'cuda'
    ):
        """
        Args:
            input_size: Number of input features
            n_classes: Number of output classes
            n_layers: Number of layers (including output)
            hidden_dim: Size of hidden layers
            weight_init_method: How to initialize weights ('zeros' or 'kaiming')
            sample_with_replacement: Whether to sample real features with replacement
            device: Device to put model on
        """
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.device = device
        
        # Build layers
        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(nn.Linear(input_size, n_classes))
        else:
            self.layers.append(nn.Linear(input_size, hidden_dim))
            for _ in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, n_classes))
            
        self.activation = nn.ReLU()
        
        # Initialize weights
        self._initialize_weights(weight_init_method)
    
    def _initialize_weights(self, method: str):
        """Initialize weights according to specified method."""
        layer = self.layers[0]
        if method == 'zeros':
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif method == 'kaiming':
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
    
    def get_first_layer_weights(self) -> torch.Tensor:
        """Returns the weights of the first layer for utility calculation."""
        return self.layers[0].weight


class FeatureRecycler:
    def __init__(
        self,
        n_features: int,
        n_real_features: int,
        distractor_chance: float,
        recycle_rate: float,
        utility_decay: float,
        use_cbp_utility: bool,
        feature_protection_steps: int,
        sample_with_replacement: bool = False,
        device: str = 'cuda'
    ):
        """
        Args:
            n_features: Total number of features model receives
            n_real_features: Number of real features available
            distractor_chance: Chance of selecting distractor vs real feature
            recycle_rate: How many features to recycle per step (can be fractional)
            utility_decay: Decay rate for feature utility
            use_cbp_utility: Whether to use CBP utility or random selection
            feature_protection_steps: Number of steps to protect new features
            device: Device to put tensors on
        """
        self.n_features = n_features
        self.n_real_features = n_real_features
        self.distractor_chance = distractor_chance
        self.recycle_rate = recycle_rate
        self.utility_decay = utility_decay
        self.use_cbp_utility = use_cbp_utility
        self.feature_protection_steps = feature_protection_steps
        self.sample_with_replacement = sample_with_replacement
        self.device = device
        
        self.recycle_accumulator = 0.0
        self.features = {}
        self._initialize_features()
        self.total_recycled = 0  # Add counter for total recycled features
    
    def _initialize_features(self):
        """Initialize the initial pool of features."""
        for i in range(self.n_features):
            self._add_new_feature(i, 0)
    
    def _add_new_feature(self, idx: int, step: int):
        """Add a new feature (real or distractor) at the given index."""
        is_real = random.random() > self.distractor_chance
        
        # Get list of currently used feature indices
        used_indices = set([
            f.distribution_params['feature_idx'] 
            for f in self.features.values() 
            if f.is_real
        ]) if not self.sample_with_replacement else set()
            
        if is_real and len(used_indices) < self.n_real_features:
            # Get available indices
            available_indices = [
                i for i in range(self.n_real_features) 
                if i not in used_indices
            ]
            
            dist_params = {
                'type': 'real',
                'feature_idx': random.choice(available_indices)
            }
        else:
            is_real = False
            
            # 50% chance of uniform vs normal distribution
            if random.random() < 0.5:
                dist_params = {
                    'type': 'distractor',
                    'distribution': 'uniform',
                    'low': random.uniform(-1, 0),
                    'high': random.uniform(0, 1)
                }
            else:
                dist_params = {
                    'type': 'distractor',
                    'distribution': 'normal',
                    'mean': random.uniform(-0.5, 0.5),
                    'std': random.uniform(0.2, 0.5)
                }
        
        self.features[idx] = FeatureInfo(
            is_real=is_real,
            utility=0.0,
            distribution_params=dist_params,
            last_update=step,
            creation_step=step,
        )
    
    def _generate_feature_values(self, batch_size: int, real_features: torch.Tensor) -> torch.Tensor:
        """Generate feature values for the current feature set."""
        values = torch.zeros(batch_size, self.n_features, device=self.device)
        
        for i in range(self.n_features):
            if self.features[i].is_real:
                feature_idx = self.features[i].distribution_params['feature_idx']
                values[:, i] = real_features[:, feature_idx]
            else:
                params = self.features[i].distribution_params
                if params['distribution'] == 'uniform':
                    values[:, i] = torch.rand(batch_size, device=self.device) * (params['high'] - params['low']) + params['low']
                else:  # normal
                    values[:, i] = torch.normal(params['mean'], params['std'], size=(batch_size,), device=self.device).clamp(-1, 1)
        
        return values
    
    def _update_utilities(
        self, 
        feature_values: torch.Tensor, 
        first_layer_weights: torch.Tensor,
        step: int
    ):
        """Update utility values for all features."""
        if not self.use_cbp_utility:
            return
            
        weight_norms = torch.norm(first_layer_weights, p=1, dim=0)
        feature_impacts = torch.abs(feature_values) * weight_norms
        
        # Update running averages
        for i in range(self.n_features):
            impact = feature_impacts[:, i].mean().item()
            old_utility = self.features[i].utility
            self.features[i].utility = (
                self.utility_decay * old_utility + 
                (1 - self.utility_decay) * impact
            )
            self.features[i].last_update = step
    
    def get_features_to_recycle(self, current_step: int) -> list:
        """Determine which features should be recycled this step."""
        self.recycle_accumulator += self.recycle_rate
        n_recycle = int(self.recycle_accumulator)
        self.recycle_accumulator -= n_recycle
        
        if n_recycle == 0:
            return []
        
        # Filter out protected features
        eligible_features = [
            i for i, f in self.features.items() 
            if current_step - f.creation_step >= self.feature_protection_steps
        ]
        
        if not eligible_features:
            return []
        
        if self.use_cbp_utility:
            utilities = {i: self.features[i].utility for i in eligible_features}
            return sorted(utilities.keys(), key=lambda x: utilities[x])[:n_recycle]
        else:
            return random.sample(eligible_features, min(n_recycle, len(eligible_features)))

    def get_statistics(self, current_step: int, model: RecyclingMLP) -> dict:
        """Calculate statistics about current features."""
        real_features = [f for f in self.features.values() if f.is_real]
        distractor_features = [f for f in self.features.values() if not f.is_real]
        
        # Get indices of real and distractor features
        real_indices = [i for i, f in self.features.items() if f.is_real]
        distractor_indices = [i for i, f in self.features.items() if not f.is_real]
        
        # Get first layer weights and calculate l1 norms
        first_layer_weights = model.get_first_layer_weights()
        weight_norms = torch.norm(first_layer_weights, p=1, dim=0)
        
        stats = {
            'avg_lifespan_real': np.mean([current_step - f.last_update for f in real_features]) if real_features else 0,
            'avg_lifespan_distractor': np.mean([current_step - f.last_update for f in distractor_features]) if distractor_features else 0,
            'num_real_features': len(real_features),
            'num_distractor_features': len(distractor_features),
            'total_recycled_features': self.total_recycled,
            'mean_weight_norm_real': weight_norms[real_indices].mean().item() if real_indices else 0,
            'mean_weight_norm_distractor': weight_norms[distractor_indices].mean().item() if distractor_indices else 0
        }
        if self.use_cbp_utility:
            stats['mean_utility_real'] = np.mean([f.utility for f in real_features]) if real_features else 0
            stats['mean_utility_distractor'] = np.mean([f.utility for f in distractor_features]) if distractor_features else 0
        
        return stats
    
    def step(
        self, 
        batch_size: int,
        real_features: torch.Tensor,
        first_layer_weights: torch.Tensor,
        step_num: int
    ) -> torch.Tensor:
        """
        Perform one step of feature recycling and return feature values.
        
        Args:
            batch_size: Size of current batch
            real_features: Real feature values for this batch
            first_layer_weights: Weights from first layer of model
            step_num: Current training step
        
        Returns:
            Tensor of feature values to use for this step
        """
        # Generate current feature values
        feature_values = self._generate_feature_values(batch_size, real_features)
        
        # Update utilities
        self._update_utilities(feature_values, first_layer_weights, step_num)
        
        # Update total recycled counter
        recycled_features = self.get_features_to_recycle(step_num)
        self.total_recycled += len(recycled_features)
        
        for idx in recycled_features:
            self._add_new_feature(idx, step_num)
        
        return feature_values


def prepare_task(args: argparse.Namespace):
    if args.task == 'dummy':
        return DummyTask(args.n_features, args.n_classes, args.task_type)
    elif args.task == 'static_linear_geoff':
        # Non-stochastic version of the 1-layer GEOFF task
        args.n_classes = 1
        args.task_type = 'regression'
        return GEOFFTask(args.n_real_features, -1, args.n_real_features)
    elif args.task == 'linear_geoff':
        # Stochastic version of the 1-layer GEOFF task
        args.n_classes = 1
        args.task_type = 'regression'
        return GEOFFTask(args.n_real_features, 20, args.n_real_features)


def set_seed(seed: Optional[int]):
    """Set random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Feature Recycling Experiment")
    
    # Dataset parameters
    parser.add_argument('--task_type', type=str, default='classification',
                      choices=['classification', 'regression'],
                      help='Type of task to perform')
    parser.add_argument('--n_classes', type=int, default=10,
                      help='Number of classes for classification')
    
    # Feature recycling parameters
    parser.add_argument('--n_features', type=int, default=100,
                      help='Total number of features model receives')
    parser.add_argument('--n_real_features', type=int, default=50,
                      help='Number of real features available')
    parser.add_argument('--distractor_chance', type=float, default=0.5,
                      help='Chance of selecting distractor vs real feature')
    parser.add_argument('--recycle_rate', type=float, default=0.1,
                      help='How many features to recycle per step')
    
    # Model parameters
    parser.add_argument('--use_cbp_utility', action='store_true',
                      help='Whether to use CBP utility for feature selection')
    parser.add_argument('--utility_decay', type=float, default=0.99,
                      help='Decay rate for feature utility')
    parser.add_argument('--weight_init_method', type=str, default='kaiming',
                      choices=['zeros', 'kaiming'],
                      help='How to initialize weights in the first layer')
    parser.add_argument('--n_layers', type=int, default=1,
                      help='Number of layers in the model')
    # Add hidden_dim parameter
    parser.add_argument('--hidden_dim', type=int, default=256,
                      help='Size of hidden layers')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--total_steps', type=int, default=10000,
                      help='Total number of training steps')
    
    # IDBD parameters (to be implemented later)
    parser.add_argument('--init_step_size', type=float, default=0.1,
                      help='Initial step size for IDBD')
    parser.add_argument('--use_idbd', action='store_true',
                      help='Whether to use IDBD')
    
    # Task parameters
    parser.add_argument('--task', type=str,
                      choices=['dummy', 'static_linear_geoff', 'linear_geoff'],
                      help='Type of task to perform')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility')
    parser.add_argument('--log_freq', type=int, default=100,
                      help='How often to log statistics')
    parser.add_argument('--feature_protection_steps', type=int, default=100,
                      help='Number of steps to protect new features')
    parser.add_argument('--wandb', action='store_true',
                      help='Whether to log to wandb')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    set_seed(args.seed)
    
    if not args.wandb:
        os.environ['WANDB_DISABLED'] = 'true'
    
    # Initialize wandb
    wandb.init(project='feature-recycling', config=vars(args))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    task = prepare_task(args)
    task_iterator = task.get_iterator(args.batch_size)
    
    # Initialize model and optimizer
    model = RecyclingMLP(
        input_size=args.n_features,
        n_classes=args.n_classes,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        weight_init_method=args.weight_init_method,
        device=device
    ).to(device)
    
    criterion = (nn.CrossEntropyLoss() if args.task_type == 'classification' 
                else nn.MSELoss())
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Initialize feature recycler
    recycler = FeatureRecycler(
        n_features=args.n_features,
        n_real_features=args.n_real_features,
        distractor_chance=args.distractor_chance,
        recycle_rate=args.recycle_rate,
        utility_decay=args.utility_decay,
        use_cbp_utility=args.use_cbp_utility,
        feature_protection_steps=args.feature_protection_steps,
        device=device
    )
    
    # Training loop
    step = 0
    pbar = tqdm(total=args.total_steps, desc="Training")
    
    # Initialize accumulators
    loss_acc = 0.0
    accuracy_acc = 0.0
    n_steps_since_log = 0
    
    while step < args.total_steps:
        # Generate batch of data
        inputs, targets = next(task_iterator)
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Get recycled features
        features = recycler.step(
            batch_size=inputs.size(0),
            real_features=inputs,
            first_layer_weights=model.get_first_layer_weights(),
            step_num=step
        )
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        loss_acc += loss.item()
        n_steps_since_log += 1
        
        # Calculate and accumulate accuracy for classification
        if isinstance(criterion, nn.CrossEntropyLoss):
            _, predicted = outputs.max(1)
            accuracy = predicted.eq(targets).float().mean().item()
            accuracy_acc += accuracy
        
        # Log metrics
        if step % args.log_freq == 0:
            metrics = {
                'step': step,
                'loss': loss_acc / n_steps_since_log,
                'accuracy': accuracy_acc / n_steps_since_log if isinstance(criterion, nn.CrossEntropyLoss) else None
            }
            # Add recycler statistics
            metrics.update(recycler.get_statistics(step, model))
            wandb.log(metrics)
            
            # Reset accumulators
            loss_acc = 0.0
            accuracy_acc = 0.0
            n_steps_since_log = 0
        
        step += 1
        pbar.update(1)
    
    pbar.close()
    wandb.finish()
