from dataclasses import dataclass
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import hydra
import omegaconf
from omegaconf import DictConfig

from adam import Adam
from idbd import IDBD
from models import MLP
from tasks import DummyTask, GEOFFTask, NonlinearGEOFFTask


@dataclass
class FeatureInfo:
    """Stores information about a feature including its utility and distribution parameters."""
    is_real: bool
    utility: float
    distribution_params: Dict[str, Any]
    last_update: int
    creation_step: int


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
        values = torch.zeros(batch_size, self.n_features)
        
        # Initialize arrays for distractor indices
        uniform_indices = []
        normal_indices = []
        uniform_lows = []
        uniform_highs = []
        normal_means = []
        normal_stds = []
        
        # Single loop to handle all features
        for i in range(self.n_features):
            if self.features[i].is_real:
                feature_idx = self.features[i].distribution_params['feature_idx']
                values[:, i] = real_features[:, feature_idx]
            else:
                params = self.features[i].distribution_params
                if params['distribution'] == 'uniform':
                    uniform_indices.append(i)
                    uniform_lows.append(params['low'])
                    uniform_highs.append(params['high'])
                else:  # normal
                    normal_indices.append(i)
                    normal_means.append(params['mean'])
                    normal_stds.append(params['std'])
        
        # Handle uniform distractors in batch
        if uniform_indices:
            lows = torch.tensor(uniform_lows)
            highs = torch.tensor(uniform_highs)
            
            uniform_values = torch.rand(batch_size, len(uniform_indices))
            uniform_values = uniform_values * (highs - lows) + lows
            values[:, uniform_indices] = uniform_values
            
        # Handle normal distractors in batch
        if normal_indices:
            means = torch.tensor(normal_means)
            stds = torch.tensor(normal_stds)
            
            eps = torch.randn(batch_size, len(normal_indices))
            normal_values = (means + eps * stds).clamp(-1, 1)
            values[:, normal_indices] = normal_values
            
        values = values.to(self.device)
        
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
        feature_impacts = feature_impacts.mean(dim=0).detach().cpu().numpy()
        old_utilities = np.array([self.features[i].utility for i in range(self.n_features)])
        new_utilities = self.utility_decay * old_utilities + (1 - self.utility_decay) * feature_impacts
        
        # Update running averages
        for i in range(self.n_features):
            self.features[i].utility = new_utilities[i]
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

    def get_statistics(self, current_step: int, model: MLP, optimizer: optim.Optimizer) -> dict:
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
            'avg_lifespan_real': np.mean([current_step - f.creation_step for f in real_features]) if real_features else 0,
            'avg_lifespan_distractor': np.mean([current_step - f.creation_step for f in distractor_features]) if distractor_features else 0,
            'num_real_features': len(real_features),
            'num_distractor_features': len(distractor_features),
            'total_recycled_features': self.total_recycled,
            'mean_weight_norm_real': weight_norms[real_indices].mean().item() if real_indices else 0,
            'mean_weight_norm_distractor': weight_norms[distractor_indices].mean().item() if distractor_indices else 0
        }
        
        if self.use_cbp_utility:
            stats['mean_utility_real'] = np.mean([f.utility for f in real_features]) if real_features else 0
            stats['mean_utility_distractor'] = np.mean([f.utility for f in distractor_features]) if distractor_features else 0
            
        if isinstance(optimizer, IDBD):
            first_layer = model.layers[0]
            idbd_beta = optimizer.state[first_layer.weight]['beta']
            learning_rates = torch.exp(idbd_beta).mean(dim=0)
            stats['mean_learning_rate_real'] = learning_rates[real_indices].mean().item() if real_indices else 0
            stats['mean_learning_rate_distractor'] = learning_rates[distractor_indices].mean().item() if distractor_indices else 0
            
        elif isinstance(optimizer, Adam):
            first_layer = model.layers[0]
            state = optimizer.state[first_layer.weight]
            
            # Get Adam parameters
            step = state['step']
            exp_avg_sq = state['exp_avg_sq']
            beta1, beta2 = optimizer.defaults['betas']
            lr = optimizer.defaults['lr']
            eps = optimizer.defaults['eps']
            
            # Calculate bias corrections
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            
            # Calculate step size
            step_size = lr / bias_correction1
            
            # Calculate denominator
            denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)
            
            # Calculate effective learning rates
            effective_lrs = (step_size / denom).mean(dim=0)
            
            stats['mean_learning_rate_real'] = effective_lrs[real_indices].mean().item() if real_indices else 0
            stats['mean_learning_rate_distractor'] = effective_lrs[distractor_indices].mean().item() if distractor_indices else 0
        
        return stats
    
    def step(
        self, 
        batch_size: int,
        real_features: torch.Tensor,
        first_layer_weights: torch.Tensor,
        step_num: int
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Perform one step of feature recycling and return feature values.
        
        Args:
            batch_size: Size of current batch
            real_features: Real feature values for this batch
            first_layer_weights: Weights from first layer of model
            step_num: Current training step
        
        Returns:
            Tensor of feature values to use for this step
            List of indices of features that were recycled
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
        
        return feature_values, recycled_features


def reset_feature_weights(idxs: Union[int, Sequence[int]], model: MLP, optimizer: optim.Optimizer, cfg: DictConfig):
    """Reset the weights and associated optimizer state for a feature."""
    if isinstance(idxs, Sequence) and len(idxs) == 0:
        return
    
    first_layer = model.layers[0]
    
    # Reset weights
    if cfg.model.weight_init_method == 'zeros':
        with torch.no_grad():
            first_layer.weight[:, idxs] = 0
    elif cfg.model.weight_init_method == 'kaiming_uniform':
        fan = first_layer.weight.shape[1] # fan_in
        gain = 1
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            first_layer.weight[:, idxs] = first_layer.weight[:, idxs].uniform_(-bound, bound)
    else:
        raise ValueError(f'Invalid weight initialization method: {cfg.model.weight_init_method}')

    # Reset optimizer states
    if isinstance(optimizer, Adam):
        # Reset Adam state for the specific feature
        state = optimizer.state[first_layer.weight]
        if len(state) > 0: # State is only populated after the first call to step
            state['step'][:, idxs] = 0
            state['exp_avg'][:, idxs] = 0
            state['exp_avg_sq'][:, idxs] = 0
            if 'max_exp_avg_sq' in state:  # For AMSGrad
                state['max_exp_avg_sq'][:, idxs] = 0
    elif isinstance(optimizer, IDBD):
        state = optimizer.state[first_layer.weight]
        state['beta'][:, idxs] = math.log(cfg.train.learning_rate)
        state['h'][:, idxs] = 0
    else:
        raise ValueError(f'Invalid optimizer type: {type(optimizer)}')


def prepare_task(cfg: DictConfig):
    """Prepare the task based on configuration."""
    if cfg.task.name.lower() == 'dummy':
        return DummyTask(cfg.task.n_features, cfg.model.output_dim, cfg.task.type)
    elif cfg.task.name.lower() == 'static_linear_geoff':
        # Non-stochastic version of the 1-layer GEOFF task
        cfg.model.output_dim = 1
        cfg.task.type = 'regression'
        return GEOFFTask(cfg.task.n_real_features, -1, cfg.task.n_real_features, seed=cfg.seed)
    elif cfg.task.name.lower() == 'linear_geoff':
        # Stochastic version of the 1-layer GEOFF task
        cfg.model.output_dim = 1
        cfg.task.type = 'regression'
        return GEOFFTask(cfg.task.n_real_features, 20, cfg.task.n_real_features, seed=cfg.seed)
    elif cfg.task.name.lower() == 'nonlinear_geoff':
        cfg.model.output_dim = 1
        cfg.task.type = 'regression'
        return NonlinearGEOFFTask(
            n_features=cfg.task.n_real_features,
            flip_rate=cfg.task.flip_rate,
            n_layers=cfg.task.n_layers,
            hidden_dim=cfg.task.hidden_dim,
            weight_scale=cfg.task.weight_scale,
            activation=cfg.task.activation,
        )


def prepare_optimizer(model: nn.Module, cfg: DictConfig):
    """Prepare the optimizer based on configuration."""
    if cfg.train.optimizer == 'adam':
        return Adam(
            model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
    elif cfg.train.optimizer == 'rmsprop':
        return Adam(
            model.parameters(),
            lr=cfg.train.learning_rate,
            betas=(0, 0.999),
            weight_decay=cfg.train.weight_decay,
        )
    elif cfg.train.optimizer == 'idbd':
        return IDBD(
            model.parameters(),
            init_lr=cfg.train.learning_rate,
            meta_lr=cfg.idbd.meta_learning_rate,
            trace_diagonal_approx=cfg.idbd.diagonal_approx,
            weight_decay=cfg.train.weight_decay,
        )


def set_seed(seed: Optional[int]):
    """Set random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@hydra.main(config_path='conf', config_name='defaults')
def main(cfg: DictConfig) -> None:
    """Run the feature recycling experiment."""
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True)
        
    set_seed(cfg.seed)
    
    if not cfg.wandb:
        os.environ['WANDB_DISABLED'] = 'true'
    
    # Initialize wandb
    wandb.init(project=cfg.project)
    
    task = prepare_task(cfg)
    task_iterator = task.get_iterator(cfg.train.batch_size)
    
    # Initialize model and optimizer
    model = MLP(
        input_dim=cfg.task.n_features,
        output_dim=cfg.model.output_dim,
        n_layers=cfg.model.n_layers,
        hidden_dim=cfg.model.hidden_dim,
        weight_init_method=cfg.model.weight_init_method,
        activation=cfg.model.activation,
        device=cfg.device
    ).to(cfg.device)
    
    criterion = (nn.CrossEntropyLoss() if cfg.task.type == 'classification' 
                else nn.MSELoss())
    optimizer = prepare_optimizer(model, cfg)
    
    # Initialize feature recycler
    recycler = FeatureRecycler(
        n_features=cfg.task.n_features,
        n_real_features=cfg.task.n_real_features,
        distractor_chance=cfg.feature_recycling.distractor_chance,
        recycle_rate=cfg.feature_recycling.recycle_rate,
        utility_decay=cfg.feature_recycling.utility_decay,
        use_cbp_utility=cfg.feature_recycling.use_cbp_utility,
        feature_protection_steps=cfg.feature_recycling.feature_protection_steps,
        device=cfg.device
    )
    
    # Training loop
    step = 0
    pbar = tqdm(total=cfg.train.total_steps, desc='Training')
    
    # Initialize accumulators
    cumulative_loss = np.float128(0.0)
    loss_acc = 0.0
    accuracy_acc = 0.0
    n_steps_since_log = 0
    
    while step < cfg.train.total_steps:
        # Generate batch of data
        inputs, targets = next(task_iterator)
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
        
        # Get recycled features
        features, recycled_features = recycler.step(
            batch_size=inputs.size(0),
            real_features=inputs,
            first_layer_weights=model.get_first_layer_weights(),
            step_num=step
        )

        # Reset weights and optimizer states for recycled features
        reset_feature_weights(recycled_features, model, optimizer, cfg)
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward(create_graph=isinstance(optimizer, IDBD))
        optimizer.step()
        
        # Accumulate metrics
        loss_acc += loss.item()
        cumulative_loss += loss.item()
        n_steps_since_log += 1
        
        # Calculate and accumulate accuracy for classification
        if isinstance(criterion, nn.CrossEntropyLoss):
            _, predicted = outputs.max(1)
            accuracy = predicted.eq(targets).float().mean().item()
            accuracy_acc += accuracy
        
        # Log metrics
        if step % cfg.train.log_freq == 0:
            metrics = {
                'step': step,
                'samples': step * cfg.train.batch_size,
                'loss': loss_acc / n_steps_since_log,
                'cumulative_loss': cumulative_loss,
                'accuracy': accuracy_acc / n_steps_since_log if isinstance(criterion, nn.CrossEntropyLoss) else None,
                'squared_targets': targets.square().mean().item(),
            }
            # Add recycler statistics
            metrics.update(recycler.get_statistics(step, model, optimizer))
            wandb.log(metrics)
            
            pbar.set_postfix(loss=metrics['loss'], accuracy=metrics['accuracy'])
            
            # Reset accumulators
            loss_acc = 0.0
            accuracy_acc = 0.0
            n_steps_since_log = 0
        
        step += 1
        pbar.update(1)
    
    pbar.close()
    wandb.finish()


if __name__ == '__main__':
    main()
