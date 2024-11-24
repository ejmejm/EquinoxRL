import math

import torch
from torch.optim.optimizer import Optimizer
from typing import Iterator, Optional


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
        trace_diagonal_approx: bool = True,
    ):
        defaults = dict(meta_lr=meta_lr)
        super().__init__(params, defaults)
        self.trace_diagonal_approx = trace_diagonal_approx

        # Initialize beta and h for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['beta'] = torch.full_like(p.data, math.log(init_lr))
                state['h'] = torch.zeros_like(p.data)
    

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            Optional computed loss from closure
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # If I want to do a faithful deep IDBD implementation, then the right-hand-side of equation 12
        # needs to be alpha * second order derivative of the loss with respect to beta

        param_updates = []
        for group in self.param_groups:
            meta_lr = group['meta_lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                state = self.state[p]
                
                # Get state variables
                beta = state['beta']
                h = state['h']
                
                # Update beta
                beta.add_(meta_lr * grad * h)
                
                # Calculate alpha (learning rate)
                alpha = torch.exp(beta)
                
                # Queue paramter update
                param_update = -alpha * grad
                param_updates.append((p, param_update))
                
                # Calculate second order grad for h
                try:
                    if self.trace_diagonal_approx:
                        # Create a diagonal mask to extract only diagonal elements of the Hessian
                        mask = torch.eye(grad.numel()).reshape(grad.numel(), *grad.shape)
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
                state['h'] = h * (1 - alpha * second_order_grad).clamp(min=0) + alpha * grad
                
                # Store updated states
                state['beta'] = beta
                
                # print(f'Alpha: [{alpha[0][0].item():.4f}, {alpha[0][1].item():.4f}] | '
                #       f'h: [{h[0][0].item():.2f}, {h[0][1].item():.2f}] | '
                #       f'grad: [{grad[0][0].item():.2f}, {grad[0][1].item():.2f}] | '
                #       f'param: [{p[0][0].item():.2f}, {p[0][1].item():.2f}]')
                
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