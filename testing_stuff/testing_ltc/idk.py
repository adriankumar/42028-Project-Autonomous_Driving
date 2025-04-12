import torch
import torch.nn as nn

# Define dimensions for testing
input_dim = 50  # Example value
hidden_dim = 20  # Example value

# Create parameters like in your class
# sensory_mu = nn.Parameter(torch.rand(input_dim, hidden_dim) * 0.5 + 0.3)
# sensory_sigma = nn.Parameter(torch.rand(input_dim, hidden_dim) * 5.0 + 3.0)
# sensory_weight = nn.Parameter(torch.rand(input_dim, hidden_dim) * (1.0 - 0.01) + 0.01)
# sensory_reversal = nn.Parameter(2 * (torch.randint(0, 2, (input_dim, hidden_dim), dtype=torch.float32)) - 1)

# # Print the shapes of the parameters
# print(f"sensory_mu shape: {sensory_mu.shape}")
# print(f"sensory_sigma shape: {sensory_sigma.shape}")
# print(f"sensory_weight shape: {sensory_weight.shape}")
# print(f"sensory_reversal shape: {sensory_reversal.shape}")

# # Also print some sample values to verify the ranges
# print(f"\nSample values from sensory_mu (should be ~Uniform(0.3, 0.8)): {sensory_mu[0, 0:5]}")
# print(f"Sample values from sensory_sigma (should be ~Uniform(3, 8)): {sensory_sigma[0, 0:5]}")
# print(f"Sample values from sensory_weight (should be ~Uniform(0.01, 1.0)): {sensory_weight[0, 0:5]}")



idk = (2 * torch.randint(0, 2, (50, 20)) - 1)
print(idk)
print(idk.shape)