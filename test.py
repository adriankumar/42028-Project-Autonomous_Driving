import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from comma_ai_processing import CommaAILTCDataset, preprocess_comma_ai_for_ltc
from LTC.ltc_model import LTC
from LTC.neural_wiring import NeuralCircuitPolicy as ncp
from LTC.model_visualisation import view_neural_wiring, plot_adjacency_matrices

#train ltc model on comma ai data
def train_ltc_model(inputs, targets, model, device, epochs=10, batch_size=8, lr=0.001, sequence_length=100, stride=50):
    #create dataset and dataloader
    dataset = CommaAILTCDataset(inputs, targets, sequence_length=sequence_length, stride=stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    #setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    #training loop
    for epoch in range(epochs):
        epoch_loss = 0
        
        for batch_inputs, batch_targets in dataloader:
            #move data to device
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            #forward pass
            outputs, _ = model(batch_inputs)
            
            #calculate loss
            loss = criterion(outputs, batch_targets)
            
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}")
    
    return losses

#evaluate model and plot results
def evaluate_model(inputs, targets, model, device, sequence_length=200):
    #limit to available data
    if sequence_length > len(inputs):
        sequence_length = len(inputs)
    
    #prepare input
    eval_inputs = torch.tensor(inputs[:sequence_length], dtype=torch.float32).unsqueeze(0).to(device)
    
    #get predictions
    with torch.no_grad():
        predictions, _ = model(eval_inputs)
    
    #convert to numpy
    predictions = predictions.squeeze().cpu().numpy()
    true_values = targets[:sequence_length]
    
    #plot results
    plt.figure(figsize=(12, 6))
    plt.plot(true_values, label='true steering angle')
    plt.plot(predictions, label='predicted steering angle')
    plt.xlabel('frame')
    plt.ylabel('steering angle')
    plt.legend()
    plt.title('ltc model steering angle prediction')
    plt.show()
    
    return predictions, true_values

#main function to run training
def main():
    #setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #neural wiring parameters
    input_dim = 20
    outgoing_sensory = 5
    num_inter = 16
    outgoing_inter = 3
    num_command = 6
    num_reccurrent = 2
    outgoing_command = 5
    output_dim = 1
    seed = 24573471
    
    #initialise wire object
    wire = ncp(
        inter_neurons=num_inter,
        command_neurons=num_command,
        motor_neurons=output_dim,
        outgoing_sensory_neurons=outgoing_sensory,
        outgoing_inter_neurons=outgoing_inter,
        num_of_recurrent_connections=num_reccurrent,
        outgoing_command_neurons=outgoing_command,
        seed=seed
    )
    
    wire.build(input_dim)
    
    #create ltc model
    model = LTC(
        wiring=wire,
        return_sequences=True,
        input_mapping="affine",
        output_mapping="affine",
        ode_unfolds=6,
        implicit_constraints=True
    )

    #move model to device
    model.to(device)
    plot_adjacency_matrices(wire, save_path="adjacency_matrices.png") #will save as image instead of popping up as window
    view_neural_wiring(wire, save_path="LTC_neural_structure.png", show=False) #will save as image instead of popping up as window
    print("saved architecture images")
    
    #paths
    camera_file = "datasets\\Comma_ai_dataset\\train\\camera\\2016-06-08--11-46-01.h5"
    label_file = "datasets\\Comma_ai_dataset\\train\\labels\\2016-06-08--11-46-01.h5"
    
    #preprocess data
    print("preprocessing data...")
    inputs, targets = preprocess_comma_ai_for_ltc(camera_file, label_file, feature_dim=input_dim, start_idx=2000, end_idx=5000)
    print(f"processed {len(inputs)} frames")

    #train model
    print("training model...")
    losses = train_ltc_model(inputs, targets, model, device, epochs=5, batch_size=4, sequence_length=100, stride=50)
    
    #evaluate model
    print("evaluating model...")
    evaluate_model(inputs, targets, model, device, sequence_length=200)

if __name__ == "__main__":
    main()