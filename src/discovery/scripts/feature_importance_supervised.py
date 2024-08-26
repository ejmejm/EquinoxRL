# Explanation of the experiment:
# The goal of this experiment is to see how the improvement rate of a model on a simple
# supervised learning task scales with the number of distractor inputs.
# Or in other words, as we add more input features that are randomly sampled from a [0, 1] uniform distribution,
# how much slower does the model learn?

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import random
import numpy as np
import argparse


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, zero_init=False, num_classes=10):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        first_layer = nn.Linear(input_size, hidden_sizes[0])
        self.layers.append(first_layer)
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        self.gelu = nn.GELU()
        
        if zero_init:
            nn.init.zeros_(first_layer.weight)
            nn.init.zeros_(first_layer.bias)
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.gelu(layer(x))
        return self.layers[-1](x)


def load_and_preprocess_data():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the image
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    return train_dataset, test_dataset


def create_dataloaders(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def initialize_model_and_optimizer(input_size, hidden_sizes, num_classes, learning_rate, weight_decay, zero_init, device):
    model = MLP(input_size=input_size, hidden_sizes=hidden_sizes, zero_init=zero_init, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return model, criterion, optimizer


def train_and_test(
        model, train_loader, test_loader, criterion, optimizer, num_distractors, total_steps,
        test_interval, log_interval, use_half_precision=False, device='cuda',
    ):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    step = 0
    epoch = 0
    samples = 0
    
    # Set up scaler for half precision training
    scaler = torch.amp.GradScaler(device) if use_half_precision else None
    
    pbar = tqdm(total=total_steps, desc="Training Progress")
    while step < total_steps:
        epoch += 1
        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)
            distractors = torch.rand(images.size(0), num_distractors, device=device)
            inputs = torch.cat((images, distractors), dim=1)
            labels = labels.to(device)
            samples += images.size(0)
            
            # Use autocast for half precision training
            with torch.amp.autocast(device, enabled=use_half_precision):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            
            if use_half_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            step += 1
            pbar.update(1)
            
            if step % log_interval == 0:
                wandb.log({
                    "step": step,
                    "epoch": epoch,
                    "train/loss": running_loss / 100,
                    "train/accuracy": 100. * correct / total
                })
                running_loss = 0.0
                correct = 0
                total = 0
            
            if step % test_interval == 0:
                test_loss, test_accuracy = test_model(model, test_loader, criterion, num_distractors, use_half_precision, device)
                wandb.log({
                    "step": step,
                    "epoch": epoch,
                    "samples": samples,
                    "test/loss": test_loss,
                    "test/accuracy": test_accuracy
                })
                model.train()
                # Log histograms and norms of the first layer weights
                first_layer_weights = model.layers[0].weight
                input_size = images.size(1)
                real_weights = first_layer_weights[:, :input_size].flatten()
                distractor_weights = first_layer_weights[:, input_size:].flatten()
                
                wandb.log({
                    "step": step,
                    "epoch": epoch,
                    "samples": samples,
                    "weights/real_input": wandb.Histogram(real_weights.detach().cpu().numpy()),
                    "weights/distractor_input": wandb.Histogram(distractor_weights.detach().cpu().numpy()),
                    "norms/real_input_l1": (torch.norm(real_weights, p=1) / real_weights.numel()).item(),
                    "norms/real_input_l2": (torch.norm(real_weights, p=2) / torch.sqrt(torch.tensor(real_weights.numel()))).item(),
                    "norms/distractor_input_l1": (torch.norm(distractor_weights, p=1) / distractor_weights.numel()).item(),
                    "norms/distractor_input_l2": (torch.norm(distractor_weights, p=2) / torch.sqrt(torch.tensor(distractor_weights.numel()))).item()
                })
            if step >= total_steps:
                break
    pbar.close()


def test_model(model, test_loader, criterion, num_distractors, use_half_precision=False, device='cuda'):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.view(images.size(0), -1).to(device)
            distractors = torch.rand(images.size(0), num_distractors, device=device)
            inputs = torch.cat((images, distractors), dim=1)
            labels = labels.to(device)
            
            if use_half_precision:
                inputs = inputs.half()
            
            with torch.amp.autocast(device, enabled=use_half_precision):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    return test_loss / len(test_loader), 100. * test_correct / test_total


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = argparse.ArgumentParser(description="Distractor Experiment")
    parser.add_argument("--num_distractors", type=int, default=0, help="Number of distractor inputs")
    parser.add_argument("--hidden_sizes", type=str, default="[256, 256]", help="Hidden layer sizes as a string representation of a list")
    parser.add_argument("--total_steps", type=int, default=10000, help="Total number of training steps")
    parser.add_argument("--log_interval", type=int, default=25, help="Log interval")
    parser.add_argument("--test_interval", type=int, default=500, help="Test interval")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--use_half_precision", type=bool, default=False, help="Use half precision training")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--zero_init", type=bool, default=False, help="Zero initialize the first layer")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    args.input_size = 16 * 16
    args.num_classes = 10
    args.hidden_sizes = eval(args.hidden_sizes)  # Convert string representation to actual list
    assert isinstance(args.hidden_sizes, list), "Hidden sizes must be a list!"
    
    return args


if __name__ == '__main__':
    args = parse_arguments()
    # Print the arguments
    print(args)

    # Set seed for reproducibility
    set_seed(args.seed)

    # Main execution
    wandb.init(project='distractor-experiment', config=vars(args))
    
    train_dataset, test_dataset = load_and_preprocess_data()
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, args.batch_size)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model, criterion, optimizer = initialize_model_and_optimizer(
        args.input_size + args.num_distractors, 
        args.hidden_sizes, 
        args.num_classes, 
        args.learning_rate,
        args.weight_decay,
        args.zero_init,
        device
    )

    train_and_test(
        model, train_loader, test_loader, criterion, optimizer, args.num_distractors,
        args.total_steps, args.test_interval, args.log_interval, args.use_half_precision,
        device,
    )

    print("Training complete.")