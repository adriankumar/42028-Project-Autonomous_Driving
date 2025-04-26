
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from LTC import LTC
from wiring_class import NeuralCircuitPolicy
from timeseries_data import X_train, X_test, y_train, y_test
from neural_wiring_visualisation import plot_wiring

#check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")

#convert numpy arrays to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

print(f"x_train tensor shape: {X_train_tensor.shape}")  #[samples, timesteps, features]
print(f"y_train tensor shape: {y_train_tensor.shape}")  #[samples]

#hyperparameters
input_size = 1  #single feature (sine wave value)
batch_size = 32
learning_rate = 0.01
num_epochs = 20


#base neuron counts
inter_neurons = 25
command_neurons = 30
motor_neurons = 1 #output is one cos we are simply predicting the next value

#connection parameters (safely within constraints), change the explicit integers for modifying constraints
sensory_fanout = min(15, inter_neurons) #cannot not exceed inter_neurons
inter_fanout = min(15, command_neurons) #cannot not exceed command_neurons
recurrent_connections = 25 #cn be any positive number
motor_fanin = min(10, command_neurons) #must not exceed command_neurons

#create ncp wiring
wiring = NeuralCircuitPolicy(
    inter_neurons=inter_neurons,      
    command_neurons=command_neurons,     
    motor_neurons=motor_neurons,
    outgoing_sensory_neurons=sensory_fanout, 
    outgoing_inter_neurons=inter_fanout,   
    num_of_recurrent_connections=recurrent_connections, 
    incoming_command_neurons=motor_fanin  
)

#build wiring with input size
wiring.build(input_size)

#visualise the wiring architecture
plot_wiring(wiring, save_path='ltc_network_architecture.png')

print("network visualisations saved")

#create ltc model
model = LTC(
    input_size=input_size,
    wiring=wiring,
    return_sequences=False, #only return final prediction
    batch_first=True,  #batch is first dimension
    input_mapping="affine",
    output_mapping="affine",
    ode_unfolds=6,
    epsilon=1e-8,
    implicit_constraints=True).to(device)

print(f"model created with {model.synapse_count} internal neurons and {model.sensory_synapse_count} sensory neurons")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"total trainable parameters: {total_params}")


#loss function and optimiser
criterion = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

#for tracking metrics
train_losses = []
test_losses = []

#training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    #create mini-batches
    permutation = torch.randperm(X_train_tensor.size(0))
    num_batches = X_train_tensor.size(0) // batch_size
    
    for i in range(num_batches):
        indices = permutation[i*batch_size:(i+1)*batch_size]
        
        batch_x = X_train_tensor[indices]
        batch_y = y_train_tensor[indices]
        
        #zero gradients
        optimiser.zero_grad()
        
        #forward pass
        outputs, _ = model(batch_x)
        outputs = outputs.squeeze() #remove extra dimensions
        
        #calculate loss
        loss = criterion(outputs, batch_y)
        
        #backward pass and optimise
        loss.backward()
        optimiser.step()
        
        epoch_loss += loss.item()
    
    avg_epoch_loss = epoch_loss / num_batches
    train_losses.append(avg_epoch_loss)
    
    #evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs, _ = model(X_test_tensor)
        test_outputs = test_outputs.squeeze()
        test_loss = criterion(test_outputs, y_test_tensor)
        test_losses.append(test_loss.item())
    
    #print progress
    if (epoch+1) % 5 == 0:
        print(f"epoch [{epoch+1}/{num_epochs}], train loss: {avg_epoch_loss:.6f}, test loss: {test_loss.item():.6f}")

#visualise training progress
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='training loss')
plt.plot(test_losses, label='test loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.title('training and validation loss')

#generate predictions on test data
model.eval()
with torch.no_grad():
    test_predictions, _ = model(X_test_tensor)
    test_predictions = test_predictions.squeeze().cpu().numpy()

#plot predictions vs actual values
plt.subplot(1, 2, 2)
test_actual = y_test.flatten()
plt.plot(test_actual, label='actual')
plt.plot(test_predictions, label='predicted')
plt.xlabel('time step')
plt.ylabel('value')
plt.legend()
plt.title('ltc model predictions')

plt.tight_layout()
plt.savefig('ltc_results.png')
plt.show()

#calculate metrics
mse = np.mean((test_predictions - test_actual) ** 2)
mae = np.mean(np.abs(test_predictions - test_actual))
print(f"test mse: {mse:.6f}")
print(f"test mae: {mae:.6f}")