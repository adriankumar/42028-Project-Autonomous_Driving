import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#set random seed for reproducibility
np.random.seed(42)

#generate time series data: sine wave with extreme noise
def generate_sine_wave_dataset(n_samples=1000, seq_length=50):
    time = np.linspace(0, 20 * np.pi, n_samples) #10 complete sine cycles
    
    #generate sine wave with increasing frequency
    frequency_factor = 1 + time / (20 * np.pi)  #frequency increases over time
    sine_wave = np.sin(frequency_factor * time)
    
    #add noise
    noise = np.random.normal(0, 0.2, n_samples)
    
    #add random spikes and drops
    spikes = np.zeros(n_samples)
    spike_locations = np.random.choice(range(n_samples), size=int(n_samples*0.05), replace=False)
    spikes[spike_locations] = np.random.uniform(0.5, 1.5, size=len(spike_locations))
    
    #add sudden phase shifts at random points
    phase_shifts = np.zeros(n_samples)
    shift_points = np.random.choice(range(n_samples//4, n_samples), size=5, replace=False)
    
    for point in shift_points:
        phase_shifts[point:point+np.random.randint(20, 50)] = np.random.uniform(-1, 1)
    
    #combine all signal components
    # signal = sine_wave + noise + spikes + phase_shifts
    signal = sine_wave + noise + phase_shifts
    
    #add occasional missing data (represented as large negative values)
    missing_indices = np.random.choice(range(n_samples), size=int(n_samples*0.02), replace=False)
    signal[missing_indices] = np.nan
    
    #fill missing values with interpolation
    mask = np.isnan(signal)
    signal[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), signal[~mask])
    
    #scale to [0, 1] range
    scaler = MinMaxScaler(feature_range=(0, 1))
    signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
    print(f"length of signal: {len(signal)}, shape: {signal.shape}")
    
    #create sequences for training
    X, y = [], []
    for i in range(len(signal) - seq_length):
        X.append(signal[i:i+seq_length]) #is the wave before the next value of the sine wave
        y.append(signal[i+seq_length]) #y is the next value in the sine wave
    
    X = np.array(X)
    y = np.array(y)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    #reshape X to [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y, signal

X, y, original_signal = generate_sine_wave_dataset()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

plt.figure(figsize=(12, 6))
plt.plot(original_signal, label='Noisy Signal')
plt.title('sine wave with extreme noise')
plt.xlabel('time step')
plt.ylabel('amplitude')
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig('noisy_signal.png')
plt.close()

print(f"X_train shape: {X_train.shape}")  # [samples, time steps, features]
print(f"y_train shape: {y_train.shape}")  # [samples]
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

print("\nexample sequence:")
print(f"input sequence: {X_train[0, :5, 0]}...")  #show first 5 time steps
print(f"target value: {y_train[0]}")