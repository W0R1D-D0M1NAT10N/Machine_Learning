
import torch
import torch.nn as nn
import os

# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # First layer: 4 inputs -> 16 neurons
        self.fc1 = nn.Linear(4, 16, bias=False)
        # Second layer: 16 neurons -> 16 neurons
        self.fc2 = nn.Linear(16, 16, bias=False)
        # Output layer: 16 neurons -> 4 outputs
        self.fc3 = nn.Linear(16, 4, bias=False)
        
        self.sigmoid=nn.Softmax()
        # Initialize all weights to 0.25
        with torch.no_grad():
            self.fc1.weight.data.fill_(0.25)
            self.fc2.weight.data.fill_(0.25)
            self.fc3.weight.data.fill_(0.25)
    
    def forward(self, x):
        x = self.fc1(x)
        
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Initialize network, input, and target
model = SimpleNN()
input_tensor = torch.tensor([0., 1., 0., 0.])
target = torch.tensor([0., 1., 0., 0.])

# Training parameters
eta = 0.001  # Reduced learning rate for stability with deeper network
last_cost = float('inf')
tolerance = 1e-6
max_iterations = 20000  # Increased iterations for deeper network

PATH = "model_weights_2layer.pth"
if os.path.exists(PATH):
    model = SimpleNN()
    model.load_state_dict(torch.load(PATH))
    model.eval()

# Custom training loop
for iteration in range(max_iterations):
    # Forward pass
    output = model(input_tensor)
    
    # Calculate cost (MSE)
    cost = 0.5 * torch.sum((output - target) ** 2)
    
    # Check for convergence
    if iteration > 0:
        cost_diff = abs((cost.item() - last_cost) / cost.item())
        if cost_diff < tolerance:
            print(f"Converged after {iteration} iterations")
            break
    
    last_cost = cost.item()
    
    # Manual weight update using backpropagation
    with torch.no_grad():
        # Calculate gradients for each layer
        # Output layer gradients
        output_error = output - target
        fc3_gradients = torch.outer(output_error, model.fc2(model.fc1(input_tensor)))
        
        # Hidden layer 2 gradients
        hidden2_error = torch.mm(model.fc3.weight.t(), output_error.unsqueeze(1)).squeeze()
        fc2_gradients = torch.outer(hidden2_error, model.fc1(input_tensor))
        
        # Hidden layer 1 gradients
        hidden1_error = torch.mm(model.fc2.weight.t(), hidden2_error.unsqueeze(1)).squeeze()
        fc1_gradients = torch.outer(hidden1_error, input_tensor)
        
        # Update weights
        model.fc3.weight -= eta * fc3_gradients
        model.fc2.weight -= eta * fc2_gradients
        model.fc1.weight -= eta * fc1_gradients
    
    # Print progress every 100 iterations
    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Cost: {cost.item():.6f}")

# Final output
print("\nTraining complete!")
print("Final output:", output.detach().numpy())
print("\nFinal weights:")
print("Layer 1:")
print(model.fc1.weight.detach().numpy())
print("\nLayer 2:")
print(model.fc2.weight.detach().numpy())
print("\nLayer 3:")
print(model.fc3.weight.detach().numpy())
torch.save(model.state_dict(), PATH)