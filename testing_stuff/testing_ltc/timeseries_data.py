import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)



#How data is structured:
# time_series_data = [1, 2, 3, 2, 1]
# sequence_length = 2

# at i = 0
# for i in range((5 - 2))
# x.append(time_series_data[0:0+2]) # --> [1, 2]
# y.append(time_series_data[0+2]) #--> [3]

# at i = 1
# for i in range(3):
# x.append(time_series_data[1:1+2]) # --> [2, 3]
# y.append(time_series_data[1+2]) #--> [2]
#essentially X is a sliding window of values in the noisy sine wave while y is the next value in the current sequence; so X features will be the values in the sequence which is 50 of them, and they will be split into 1000 - sequence length (50) samples which is 950 



# Generate time series data: sine wave with noise
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
    print(f"length of signal: {len(signal)}, shape: {signal.shape}")
    
    # Create sequences for training
    X, y = [], []
    for i in range(len(signal) - seq_length):
        X.append(signal[i:i+seq_length]) #is the wave before the next value of the sine wave
        y.append(signal[i+seq_length]) #y is the next value in the sine wave
    
    X = np.array(X)
    y = np.array(y)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Reshape X to [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y, signal

# Generate data
X, y, original_signal = generate_sine_wave_dataset()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Visualize the time series
plt.figure(figsize=(12, 6))
plt.plot(original_signal[:], label='Original Signal')
plt.title('Sine Wave with Noise')
plt.xlabel('Time step')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

# Display dataset shape
print(f"X_train shape: {X_train.shape}")  # [samples, time steps, features]
print(f"y_train shape: {y_train.shape}")  # [samples]
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Display an example sequence
print("\nExample sequence:")
print(f"Input sequence: {X_train[0, :5, 0]}...")  # Show first 5 time steps
print(f"Target value: {y_train[0]}")