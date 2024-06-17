import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms


class MLP(nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28 * 1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()


if __name__ == "__main__":

    # Set fixed random number seed
    torch.manual_seed(42)

    # Prepare CIFAR-10 dataset
    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=10, shuffle=True, num_workers=1
    )

    # Initialize the MLP
    mlp = MLP()

    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # Run the training loop
    for epoch in range(0, 5):  # 5 epochs at maximum

        # Print epoch
        print(f"Starting epoch {epoch + 1}")

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):

            # Get inputs
            inputs, targets = data

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)

            # Compute L1 loss component
            l1_weight = 1.0
            l1_parameters = []
            for parameter in mlp.parameters():
                l1_parameters.append(parameter.view(-1))
            l1 = l1_weight * mlp.compute_l1_loss(torch.cat(l1_parameters))

            # Add L1 loss component
            loss += l1

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            minibatch_loss = loss.item()
            if i % 500 == 499:
                print(
                    "Loss after mini-batch %5d: %.5f (of which %.5f L1 loss)"
                    % (i + 1, minibatch_loss, l1)
                )
                current_loss = 0.0

    # Process is complete.
    print("Training process has finished.")
