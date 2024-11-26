import random
from typing import Iterator, Optional, Tuple

import torch


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
        seed: Optional[int] = None,
    ):
        """Initialize tracking task where target is sum of first k inputs with changing signs.
        
        Args:
            feature_dim: Number of input features (default 20)
            sign_flip_interval: How often to flip signs (in steps)
            active_features: Number of features that contribute to target (default 5)
            seed: Random seed
        """
        self.feature_dim = feature_dim
        self.sign_flip_interval = sign_flip_interval
        self.active_features = active_features
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        
        # Initialize signs randomly for active features (+1 or -1)
        self.signs = torch.randint(0, 2, (active_features,), generator=self.generator) * 2 - 1
        
        self.steps_since_flip = 0
        self.task_type = 'regression'
    
    def _randomize_sign(self):
        """Randomly flip the sign of one random feature."""
        # Choose random feature to flip
        feature_to_flip = torch.randint(0, self.active_features, (1,), generator=self.generator).item()
        # Flip its sign
        self.signs[feature_to_flip] *= -1
    
    def get_iterator(self, batch_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Get iterator that generates infinite batches of data."""
        while True:
            # Generate all features from normal distribution
            inputs = torch.randn(batch_size, self.feature_dim, generator=self.generator)
            # Calculate target using only first k features with signs
            targets = (self.signs.unsqueeze(0) * inputs[:, :self.active_features]).sum(dim=1)
            
            # Update sign flip counter and flip if needed
            self.steps_since_flip += 1
            if self.sign_flip_interval > 0 and self.steps_since_flip >= self.sign_flip_interval:
                self._randomize_sign()
                self.steps_since_flip = 0
            
            yield inputs, targets.unsqueeze(1)
