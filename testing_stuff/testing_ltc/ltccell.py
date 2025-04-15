import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# DATA GENERATION (as provided)
# -------------------------------

def generate_sine_wave_dataset(n_samples=1000, seq_length=50):
    # Create time points
    time = np.linspace(0, 20 * np.pi, n_samples)  # 10 complete sine cycles
    
    # Generate sine wave with increasing frequency
    frequency_factor = 1 + time / (20 * np.pi)  # frequency increases over time
    sine_wave = np.sin(frequency_factor * time)
    
    # Add some noise
    noise = np.random.normal(0, 0.1, n_samples)
    signal = sine_wave + noise
    
    # Scale to [0, 1] range
    scaler = MinMaxScaler(feature_range=(0, 1))
    signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
    print(f"Length of signal: {len(signal)}, shape: {signal.shape}")
    
    # Create sequences: each X is a sliding window of length seq_length, y is the next value.
    X, y = [], []
    for i in range(len(signal) - seq_length):
        X.append(signal[i:i+seq_length])
        y.append(signal[i+seq_length])
    
    X = np.array(X)
    y = np.array(y)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Reshape X to [samples, time steps, features] (here features=1)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y, signal

# Generate dataset
np.random.seed(42)
X, y, original_signal = generate_sine_wave_dataset()

# Split the data (we will use this later for full model integration)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Visualize the time series
plt.figure(figsize=(12, 6))
plt.plot(original_signal, label='Original Signal')
plt.title('Sine Wave with Noise')
plt.xlabel('Time step')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

print(f"X_train shape: {X_train.shape}")  # [samples, time steps, features]
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

print("\nExample sequence:")
print(f"Input sequence (first 5 steps): {X_train[0, :5, 0]} ...")
print(f"Target value: {y_train[0]}")

# -------------------------------
# LTC CELL IMPLEMENTATION IN PYTORCH
# -------------------------------

class LTC_Cell(nn.Module):
    """
    A PyTorch implementation of a Liquid Time-Constant (LTC) cell 
    that uses a fused (semi-implicit Euler) solver.
    
    The cell models continuous-time dynamics with adaptive time-constants.
    
    Key variables:
      - input_dim: Dimensionality of the external input.
      - hidden_dim: Number of neurons in the LTC cell.
      - solver_steps: Number of fused (sub-)solver steps per time step.
      - time_constant: (cm_t) Membrane capacitance-like parameter that modulates state update.
      - leak_conductance: (gleak) Determines how fast the cell leaks toward its leak potential.
      - leak_potential: (vleak) The baseline membrane (leak) potential.
      
    Sensory (input) parameters:
      - input_weight & input_bias: For an affine mapping of the raw input.
      - sensory_mu, sensory_sigma: Parameters for the sigmoid gating of the input.
      - sensory_weight: Scales the input contribution.
      - sensory_reversal: Reversal potential for the sensory input (affects sign/direction).
      
    Recurrent (hidden-to-hidden) parameters:
      - recurrent_mu, recurrent_sigma: Parameters for the sigmoid gating applied to the hidden state.
      - recurrent_weight: Scales the recurrent contribution.
      - recurrent_reversal: Reversal potential for the recurrent connection.
      
    The fused solver updates the hidden state (hidden_state) as follows for each solver step:
        new_hidden = [time_constant * hidden_state + leak_conductance * leak_potential + recurrent_numer + sensory_numer] 
                     / [time_constant + leak_conductance + recurrent_denom + sensory_denom]
                     
    All summations are performed over the appropriate dimension.
    """
    def __init__(self, input_dim, hidden_dim, solver_steps=6, mapping_type='affine'):
        super(LTC_Cell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.solver_steps = solver_steps
        self.mapping_type = mapping_type  # options: 'affine', 'linear', or 'none'
        
        # --- Input mapping parameters ---
        if mapping_type == 'affine':
            # These parameters allow for a feature-wise affine transformation.
            self.input_weight = nn.Parameter(torch.ones(input_dim))
            self.input_bias = nn.Parameter(torch.zeros(input_dim))
        elif mapping_type == 'linear':
            self.input_weight = nn.Parameter(torch.ones(input_dim))
            self.input_bias = None
        else:
            self.input_weight = None
            self.input_bias = None
        
        # --- Sensory (input) parameters ---
        # Shapes: [input_dim, hidden_dim]
        self.sensory_mu = nn.Parameter(torch.rand(input_dim, hidden_dim) * 0.5 + 0.3)      # e.g. ~Uniform(0.3, 0.8)
        self.sensory_sigma = nn.Parameter(torch.rand(input_dim, hidden_dim) * 5.0 + 3.0)     # e.g. ~Uniform(3, 8)
        self.sensory_weight = nn.Parameter(torch.rand(input_dim, hidden_dim) * (1.0 - 0.01) + 0.01)
        # Sensory reversal potential: typically either +1 or -1.
        self.sensory_reversal = nn.Parameter(2 * (torch.randint(0, 2, (input_dim, hidden_dim), dtype=torch.float32)) - 1, requires_grad=True)
        
        # --- Recurrent (hidden-to-hidden) parameters ---
        # Shapes: [hidden_dim, hidden_dim]
        self.recurrent_mu = nn.Parameter(torch.rand(hidden_dim, hidden_dim) * 0.5 + 0.3)
        self.recurrent_sigma = nn.Parameter(torch.rand(hidden_dim, hidden_dim) * 5.0 + 3.0)
        self.recurrent_weight = nn.Parameter(torch.rand(hidden_dim, hidden_dim) * (1.0 - 0.01) + 0.01)
        self.recurrent_reversal = nn.Parameter(2 * (torch.randint(0, 2, (hidden_dim, hidden_dim), dtype=torch.float32)) - 1, requires_grad=True)
        
        # --- Leak and time constant parameters ---
        # These parameters are analogous to the leak conductance and membrane capacitance.
        self.leak_potential = nn.Parameter(torch.rand(hidden_dim) * 0.4 - 0.2)  # e.g., between -0.2 and 0.2
        self.leak_conductance = nn.Parameter(torch.rand(hidden_dim) * (1000 - 0.00001) + 0.00001)  # example range
        self.time_constant = nn.Parameter(torch.rand(hidden_dim) * (1000 - 0.000001) + 0.000001)    # example range
        
    def _apply_input_mapping(self, x):
        """
        Apply an affine or linear mapping to the raw input.
        
        x: Tensor of shape [batch_size, input_dim]
        Returns a tensor of the same shape.
        """
        if self.mapping_type == 'affine':
            return x * self.input_weight + self.input_bias
        elif self.mapping_type == 'linear':
            return x * self.input_weight
        else:
            return x

    def _sigmoid_gate(self, x, mu, sigma):
        """
        Apply a sigmoid function with learned mu and sigma.
        
        x: Tensor of shape [batch_size, D]
        mu, sigma: Parameters of shape [D, ...] (will be broadcasted appropriately)
        
        Returns: Tensor of shape [batch_size, D, ...]
        """
        # Unsqueeze x so that it can broadcast with mu and sigma.
        # For example, if x is [batch_size, D] and mu is [D, hidden_dim],
        # then x.unsqueeze(2) becomes [batch_size, D, 1] and the result is [batch_size, D, hidden_dim].
        return torch.sigmoid(sigma * (x.unsqueeze(2) - mu))
    
    def forward(self, input_t, hidden_state):
        """
        Perform one time-step update using the fused (semi-implicit Euler) solver.
        
        input_t: External input at the current time step; shape [batch_size, input_dim]
        hidden_state: Current hidden state; shape [batch_size, hidden_dim]
        
        Returns:
          updated_hidden: New hidden state after solver_steps updates; shape [batch_size, hidden_dim]
        """
        # ----- Step 1: Map the input -----
        mapped_input = self._apply_input_mapping(input_t)  # shape: [batch_size, input_dim]
        
        # ----- Step 2: Sensory (input) processing -----
        # Compute gating for sensory inputs. 
        # Resulting shape: [batch_size, input_dim, hidden_dim]
        sensory_gate = self._sigmoid_gate(mapped_input, self.sensory_mu, self.sensory_sigma)
        # Scale the gating by sensory weights.
        sensory_activation = sensory_gate * self.sensory_weight  # shape: [batch, input_dim, hidden_dim]
        # Apply reversal potential.
        sensory_reversed = sensory_activation * self.sensory_reversal  # shape: [batch, input_dim, hidden_dim]
        # Sum over the input dimension to obtain aggregated contributions.
        sensory_numerator = sensory_reversed.sum(dim=1)      # shape: [batch_size, hidden_dim]
        sensory_denominator = sensory_activation.sum(dim=1)    # shape: [batch_size, hidden_dim]
        
        # ----- Step 3: Fused solver update loop -----
        # Initialize new_hidden as the current hidden state.
        new_hidden = hidden_state  # shape: [batch_size, hidden_dim]
        for _ in range(self.solver_steps):
            # Recurrent processing: compute gating for the current hidden state.
            # For recurrent gating, treat hidden_state as having dimension [batch, hidden_dim]
            # We unsqueeze to [batch, hidden_dim, 1] so that it can be broadcast with parameters of shape [hidden_dim, hidden_dim].
            recurrent_gate = self._sigmoid_gate(new_hidden, self.recurrent_mu, self.recurrent_sigma)
            # Scale by the recurrent weights.
            recurrent_activation = recurrent_gate * self.recurrent_weight  # shape: [batch, hidden_dim, hidden_dim]
            # Apply recurrent reversal potential.
            recurrent_reversed = recurrent_activation * self.recurrent_reversal   # shape: [batch, hidden_dim, hidden_dim]
            # Aggregate contributions by summing over the input (first hidden dimension) for recurrent connections.
            recurrent_numerator = recurrent_reversed.sum(dim=1)     # shape: [batch, hidden_dim]
            recurrent_denominator = recurrent_activation.sum(dim=1)   # shape: [batch, hidden_dim]
            
            # Prepare parameters for element-wise operations (broadcasting over the batch dimension).
            # These parameters are vectors of shape [hidden_dim]; unsqueeze to [1, hidden_dim] for broadcasting.
            batch_time_constant = self.time_constant.unsqueeze(0)
            batch_leak_conductance = self.leak_conductance.unsqueeze(0)
            batch_leak_potential = self.leak_potential.unsqueeze(0)
            
            # Compute the numerator and denominator of the fused update.
            # The numerator sums the scaled previous hidden state, the leak term, and the sensory and recurrent inputs.
            fused_numerator = (batch_time_constant * new_hidden +
                               batch_leak_conductance * batch_leak_potential +
                               recurrent_numerator + sensory_numerator)
            
            # The denominator sums the time constant, leak conductance, and the aggregated weights.
            fused_denominator = (batch_time_constant +
                                 batch_leak_conductance +
                                 recurrent_denominator + sensory_denominator)
            
            # Update the hidden state.
            new_hidden = fused_numerator / fused_denominator
        
        return new_hidden

# -------------------------------
# DEMONSTRATION OF THE LTC CELL
# -------------------------------

# For demonstration, let’s assume:
#  - Input dimension is 1 (since our time-series is univariate)
#  - Hidden dimension is chosen to be, say, 16.
#  - We use the 'affine' input mapping.

input_dim = 1
hidden_dim = 16
solver_steps = 6

# Instantiate the LTC cell
ltc_cell = LTC_Cell(input_dim=input_dim, hidden_dim=hidden_dim, solver_steps=solver_steps, mapping_type='affine')

# For demonstration, assume a batch of 10 sequences at a single time step.
batch_size = 10
# Let input_t be a tensor of shape [batch_size, input_dim]
# For example, take the first time-step from X_train (which is numpy array of shape [num_samples, seq_length, 1])
input_t_np = X_train[:batch_size, 0, :]  # take first time step of each sample in the batch
input_t = torch.tensor(input_t_np, dtype=torch.float32)

# Initialize a hidden state (e.g., zeros) of shape [batch_size, hidden_dim]
hidden_state = torch.zeros(batch_size, hidden_dim)

# Pass the input and hidden state through the LTC cell.
updated_hidden = ltc_cell(input_t, hidden_state)

print("\n--- LTC Cell Forward Pass ---")
print(f"Input at current time step (shape {input_t.shape}):\n{input_t}")
print(f"Initial hidden state (shape {hidden_state.shape}):\n{hidden_state}")
print(f"Updated hidden state after fused solver (shape {updated_hidden.shape}):\n{updated_hidden}")
