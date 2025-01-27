import math

import torch
from torch.optim.optimizer import Optimizer
from typing import Dict, Iterator, Optional


EPSILON = 1e-8


class IDBD(Optimizer):
    """Incremental Delta-Bar-Delta optimizer.
    
    This is an implementation of the IDBD algorithm adapted for deep neural networks.
    Instead of working with input features directly, it uses gradients with respect
    to parameters and maintains separate learning rates for each parameter.
    
    Args:
        params: Iterable of parameters to optimize
        meta_lr: Meta learning rate (default: 0.01)
        init_lr: Initial learning rate (default: 0.01)
    """
    
    def __init__(
        self, 
        params: Iterator[torch.Tensor],
        meta_lr: float = 0.01,
        init_lr: float = 0.01,
        weight_decay: float = 0.0,
        version: str = 'squared_inputs', # squared_inputs, squared_grads, hvp, hessian_diagonal,
    ):
        defaults = dict(meta_lr=meta_lr)
        super().__init__(params, defaults)
        self.weight_decay = weight_decay
        self.version = version
        
        assert self.version in ['squared_inputs', 'squared_grads', 'hvp', 'hessian_diagonal'], \
            f"Invalid version: {self.version}. Must be one of: squared_inputs, squared_grads, hvp, hessian_diagonal."

        # Initialize beta and h for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['beta'] = torch.full_like(p.data, math.log(init_lr))
                state['h'] = torch.zeros_like(p.data)
    

    @torch.no_grad()
    def step(
        self,
        loss: torch.Tensor,
        predictions: torch.Tensor,
        param_inputs: Dict[torch.nn.parameter.Parameter, torch.Tensor],
        closure: Optional[callable] = None,
    ) -> Optional[float]:
        """Performs a single optimization step.
        
        Args:
            loss: Loss tensor of shape ()
            predictions: Predictions tensor of shape (batch_size, n_classes)
            param_inputs: Dictionary mapping linear layer weight parameters to their inputs
        """
        all_params = [p for group in self.param_groups for p in group['params']]
        
        if self.version == 'squared_grads':
            with torch.enable_grad():
                prediction_sum = torch.sum(predictions)
            prediction_grads = torch.autograd.grad(
                outputs = prediction_sum,
                inputs = all_params,
                retain_graph = True,
            )
            prediction_grads = {p: g for p, g in zip(all_params, prediction_grads)}
        
        # Compute gradients for all model parameters
        loss.backward(create_graph=True)

        param_updates = []
        for group in self.param_groups:
            meta_lr = group['meta_lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                if p in param_inputs:
                    assert len(param_inputs[p].shape) == 1, "Inputs must be 1D tensors"
                    inputs = param_inputs[p].unsqueeze(0)
                elif len(grad.shape) == 1:
                    inputs = torch.ones_like(grad)
                else:
                    raise ValueError(f"Parameter {p} not found in activations dictionary.")
                
                # Get state variables
                state = self.state[p]
                beta = state['beta']
                h = state['h']
                
                # Update beta
                beta.add_(meta_lr * grad * h)
                state['beta'] = beta
                
                # Calculate alpha (learning rate)
                alpha = torch.exp(beta)
                
                # Queue paramter update
                weight_decay_term = self.weight_decay * p.data if self.weight_decay != 0 else 0
                param_update = -alpha * (grad + weight_decay_term)
                param_updates.append((p, param_update))
                
                ## Different h update depending on version ##
                
                if self.version == 'squared_inputs':
                    state['h'] = h * (1 - alpha * inputs.pow(2)).clamp(min=0) + alpha * inputs
                    
                elif self.version == 'squared_grads':
                    state['h'] = h * (1 - alpha * prediction_grads[p].pow(2)).clamp(min=0) + alpha * grad

                elif self.version == 'hvp':
                    try:
                        second_order_grad = torch.autograd.grad(
                            grad, p, grad_outputs=torch.ones_like(grad), retain_graph=True)[0]
                    except RuntimeError as e:
                        if "grad and does not have a grad_fn" in str(e):
                            raise RuntimeError(
                                "Parameter grads do not have a grad_fn, which is required for IDBD. "
                                "You can fix this by calling the backward function with "
                                "create_graph=True [e.g. loss.backward(create_graph=True)]."
                            )
                        else:
                            raise e
                    state['h'] = h * (1 - alpha * second_order_grad).clamp(min=0) + alpha * grad
                    
                elif self.version == 'hessian_diagonal':
                    try:
                        # Create a diagonal mask to extract only diagonal elements of the Hessian
                        mask = torch.eye(grad.numel(), device=grad.device).reshape(grad.numel(), *grad.shape)
                        second_order_grad = torch.autograd.grad(
                            grad, p, grad_outputs=mask, is_grads_batched=True, retain_graph=True)[0]
                        second_order_grad = (second_order_grad * mask).sum(0)
                    except RuntimeError as e:
                        if "grad and does not have a grad_fn" in str(e):
                            raise RuntimeError(
                                "Parameter grads do not have a grad_fn, which is required for IDBD. "
                                "You can fix this by calling the backward function with "
                                "create_graph=True [e.g. loss.backward(create_graph=True)]."
                            )
                        else:
                            raise e
                    state['h'] = h * (1 - alpha * second_order_grad).clamp(min=0) + alpha * grad
                
        for p, param_update in param_updates:
            p.add_(param_update)

        return loss
    
    
class RMSPropIDBD(Optimizer):
    """RMSProp-based IDBD optimizer.
    
    This is an implementation of the IDBD algorithm adapted to RMSProp.
    
    Args:
        params: Iterable of parameters to optimize
        meta_lr: Meta learning rate (default: 0.01)
        init_lr: Initial learning rate (default: 0.01)
        trace_decay: Exponential decay rate (default: 0.999)
        weight_decay: Weight decay (default: 0.0)
        trace_diagonal_approx: Whether to use a diagonal approximation of the Hessian (default: True)
        ignore_norm_grad: Whether to ignore the gradient of the normalization term in the meta step-size trace update (default: False)
    """
    
    def __init__(
        self, 
        params: Iterator[torch.Tensor],
        meta_lr: float = 0.01,
        init_lr: float = 0.01,
        momentum_decay: float = 0.9,
        trace_decay: float = 0.999,
        weight_decay: float = 0.0,
        # version: str = 'squared_inputs', # momentum, squared_inputs, squared_grads, hvp, hessian_diagonal,
        use_legacy_idbd: bool = True,
        trace_diagonal_approx: bool = True,
        ignore_norm_grad: bool = True,
    ):
        defaults = dict(meta_lr=meta_lr, trace_decay=trace_decay)
        super().__init__(params, defaults)
        self.momentum_decay = momentum_decay
        self.trace_decay = trace_decay
        self.weight_decay = weight_decay
        # self.version = version
        self.trace_diagonal_approx = trace_diagonal_approx
        self.ignore_norm_grad = ignore_norm_grad
        self.use_legacy_idbd = use_legacy_idbd
        
        # Initialize beta and h for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['beta'] = torch.full_like(p.data, math.log(init_lr))
                state['h'] = torch.zeros_like(p.data)
                state['grad_square_trace'] = torch.zeros_like(p.data)
                state['momentum'] = torch.zeros_like(p.data)
                state['momentum_correction_factor'] = torch.ones_like(p.data)
                state['norm_correction_factor'] = torch.ones_like(p.data)
    

    @torch.no_grad()
    def step(self, param_inputs: Dict[torch.nn.parameter.Parameter, torch.Tensor]) -> Optional[float]:
        """Performs a single optimization step.
        
        Args:
            param_inputs: Dictionary mapping linear layer weight parameters to their inputs
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            Optional computed loss from closure
        """
        # If I want to do a faithful deep IDBD implementation, then the right-hand-side of equation 12
        # needs to be alpha * second order derivative of the loss with respect to beta

        param_updates = []
        for group in self.param_groups:
            meta_lr = group['meta_lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                if p in param_inputs:
                    assert len(param_inputs[p].shape) == 1, "Inputs must be 1D tensors"
                    inputs = param_inputs[p].unsqueeze(0)
                elif len(p.grad.shape) == 1:
                    inputs = torch.ones_like(p.grad)
                else:
                    raise ValueError(f"Parameter {p} not found in activations dictionary.")
                    
                grad = p.grad
                state = self.state[p]
                
                # Get state variables
                beta = state['beta']
                h = state['h']
                
                # Update the gradient square trace
                state['grad_square_trace'] = self.trace_decay * state['grad_square_trace'] + (1 - self.trace_decay) * grad.pow(2)
                state['norm_correction_factor'] *= self.trace_decay
                
                bias_corrected_trace = torch.sqrt(state['grad_square_trace']) / (1 - state['norm_correction_factor'])
                grad_norm_factor = 1.0 / (bias_corrected_trace + EPSILON)
                
                # Update momentum trace
                state['momentum'] = self.momentum_decay * state['momentum'] + (1 - self.momentum_decay) * grad
                state['momentum_correction_factor'] *= self.momentum_decay
                
                corrected_momentum = state['momentum'] / (1 - state['momentum_correction_factor'])

                # Update beta
                if self.use_legacy_idbd:
                    beta.add_(meta_lr * grad * h)
                else:
                    # beta.add_(meta_lr * grad * corrected_momentum)
                    # beta.add_(meta_lr * grad * corrected_momentum * grad_norm_factor)
                    beta.add_(meta_lr * grad * corrected_momentum * grad_norm_factor ** 2)
                
                state['beta'] = beta
                
                # Calculate alpha (learning rate)
                alpha = torch.exp(beta)
                
                if torch.rand(1) < 0.001:
                    print('alpha', alpha)
                
                # Queue paramter update
                weight_decay_term = self.weight_decay * p.data if self.weight_decay != 0 else 0
                param_update = -alpha * grad_norm_factor * (grad + weight_decay_term)
                param_updates.append((p, param_update))
                
                if self.use_legacy_idbd:
                    # Calculate second order grad for h
                    try:
                        if self.trace_diagonal_approx:
                            # Create a diagonal mask to extract only diagonal elements of the Hessian
                            mask = torch.eye(grad.numel(), device=grad.device).reshape(grad.numel(), *grad.shape)
                            second_order_grad = torch.autograd.grad(
                                grad, p, grad_outputs=mask, is_grads_batched=True, retain_graph=True)[0]
                            second_order_grad = (second_order_grad * mask).sum(0)
                        else:
                            second_order_grad = torch.autograd.grad(
                                grad, p, grad_outputs=torch.ones_like(grad), retain_graph=True)[0]
                    except RuntimeError as e:
                        if "grad and does not have a grad_fn" in str(e):
                            raise RuntimeError(
                                "Parameter grads do not have a grad_fn, which is required for IDBD. "
                                "You can fix this by calling the backward function with "
                                "create_graph=True [e.g. loss.backward(create_graph=True)]."
                            )
                        else:
                            raise e
                    
                    # Update h (activation trace)
                    # state['h'] = h * (1 - alpha * inputs.pow(2) * grad_norm_factor).clamp(min=0) + alpha * grad * grad_norm_factor
                    state['h'] = h * (1 - alpha * second_order_grad * grad_norm_factor).clamp(min=0) + alpha * grad * grad_norm_factor
                    
                    # state['h'] = h - alpha * grad_norm_factor * (grad + second_order_grad)
                    # state['h'] = h * (1 - alpha * grad_norm_factor * second_order_grad).clamp(min=0) + alpha * grad_norm_factor * grad
                    
                    if not self.ignore_norm_grad:
                        raise NotImplementedError("RMSPropIDBD does not support ignore_norm_grad=False.")
                        # state['r'] = ...
                        state['h'] -= alpha * grad * state['r'] / (2 * grad_norm_factor ** (1.5) * (1 - state['bias_correction_factor']))
                
        for p, param_update in param_updates:
            p.add_(param_update)
            p.grad = None

        return loss
    

if __name__ == '__main__':
    print("Testing IDBD optimizer...")
    
    # Test 1
    print("\nTest 1: Linear Regression w/ Overshooting (IDBD decreases learning rate)")
    torch.manual_seed(42)
    X = torch.tensor([[1.0, 2.0]])
    true_weights = torch.tensor([[1.5, -0.5]])
                              # [0.5, 1.0]])
    y = X @ true_weights.t()

    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.data.copy_(torch.tensor([[1.0, -1.0]]))
                                           # [0.5, 1.0]]))
    optimizer = IDBD(model.parameters(), meta_lr=0.01, init_lr=0.5)
    
    for _ in range(10):
        y_pred = model(X[0])
        loss = 0.5 * torch.nn.functional.mse_loss(y_pred, y[0])
        print('Loss:', loss)
        loss.backward(create_graph=True)
        optimizer.step()

    # Test 2
    print("\nTest 2: Linear Regression w/ Undershooting (IDBD increases learning rate)")
    with torch.no_grad():
        model.weight.data.copy_(torch.tensor([[1.0, -1.0]]))
                                           # [0.5, 1.0]]))
    optimizer = IDBD(model.parameters(), meta_lr=5.0, init_lr=0.001)
    
    for _ in range(10):
        y_pred = model(X[0])
        loss = 0.5 * torch.nn.functional.mse_loss(y_pred, y[0])
        print('Loss:', loss)
        loss.backward(create_graph=True)
        optimizer.step()
    
    
    # # Test with 2-layer network
    # print("\nTest: 2-layer network")
    # torch.manual_seed(42)
    
    # # Create random input
    # X = torch.randn(10, 4)
    # y = torch.randn(10, 2)

    # # Create 2-layer network
    # model = torch.nn.Sequential(
    #     torch.nn.Linear(4, 8),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(8, 2)
    # )
    
    # optimizer = IDBD(model.parameters(), meta_lr=0.01, init_lr=0.01)
    
    # # Single training step
    
    # for _ in range(100):
    #     y_pred = model(X)
    #     loss = torch.nn.functional.mse_loss(y_pred, y)
    #     print('Loss:', loss.item())
    #     loss.backward(create_graph=True)
    #     optimizer.step()