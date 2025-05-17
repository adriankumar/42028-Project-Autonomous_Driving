import torch
import torch.nn as nn
import numpy as np
import os
import json
import datetime
from data_preprocessing.dataset_loader import HiddenStateManager
from tqdm import tqdm
import torch.nn.functional as F

#training hyperparameters
DEFAULT_LEARNING_RATE = 0.0005
DEFAULT_CONV_GRAD_SCALING = 1.0
DEFAULT_CURVE_FACTOR = 0.0
DEFAULT_CLIP_GRAD_NORM = 10.0
DEFAULT_ACCEL_FAC = 0.15


#---------------------------
# Loss functions - these weighted loss functions were directly inspired by MIT researcher's offical 'wormnet' model training, using magnitude weighted loss
#---------------------------
#weighted loss to apply to the magnitude of prediction, i.e if steering true has a large angle 41, and model predicts small angle, we penalise this prediction more strictly as it needs to properly adjust to larger magnitudes
#sample applies for the car acceleration prediction
def magnitude_weighted_mse_loss(y_pred, y_true, mag_factor=DEFAULT_CURVE_FACTOR, accel_factor=DEFAULT_ACCEL_FAC):
    #if accel_factor not provided, use same factor for both outputs
    if accel_factor is None:
        accel_factor = mag_factor
    
    #extract steering and acceleration components
    steering_pred = y_pred[:, :, 0:1]  #shape: [batch, seq, 1]
    accel_pred = y_pred[:, :, 1:2]  #shape: [batch, seq, 1]
    
    steering_true = y_true[:, :, 0:1]  
    accel_true = y_true[:, :, 1:2]     
    
    #steering loss calculation with steering factor
    steering_weights = torch.exp(torch.abs(steering_true) * mag_factor) #create weights
    steering_squared_error = (steering_pred - steering_true) ** 2 #manual sqaured error calculation
    steering_weighted_error = steering_weights * steering_squared_error #add weight factor to error, this penalises wrong predictions with large magnitude over smaller errors, placing emphasis on larger value predictions
    steering_loss = torch.sum(steering_weighted_error) / torch.sum(steering_weights)
    
    #acceleration loss calculation with acceleration factor
    accel_weights = torch.exp(torch.abs(accel_true) * accel_factor)
    accel_squared_error = (accel_pred - accel_true) ** 2
    accel_weighted_error = accel_weights * accel_squared_error
    accel_loss = torch.sum(accel_weighted_error) / torch.sum(accel_weights)
    
    #combined loss (equal weighting)
    combined_loss = steering_loss + (1.5 * accel_loss) #scaling accel loss, since acceleration predictions seem to be learnt slower then steering angles
    
    return combined_loss

#alternative implementation which penalises later predictions over earlier ones, with the intuition that we care more about most 'recent' predictions rather than previous ones
#however this proved to make learning slow, but keep anyways, does not scale the acceleration lost because we did not use this during training for car acceleration predictions
def exponential_weighted_mse_loss(y_pred, y_true, curve_factor=DEFAULT_CURVE_FACTOR, accel_factor=None):
    #if accel_factor not provided, use same factor for both outputs
    if accel_factor is None:
        accel_factor = curve_factor
    
    seq_len = y_pred.size(1)
    
    #sequence weights tensor
    seq_weights = torch.tensor([curve_factor ** (seq_len - i - 1) for i in range(seq_len)], 
                              device=y_pred.device)
    seq_weights = seq_weights / seq_weights.sum()
    
    #extract steering and acceleration components
    steering_pred = y_pred[:, :, 0:1]  
    accel_pred = y_pred[:, :, 1:2]     
    
    steering_true = y_true[:, :, 0:1]  
    accel_true = y_true[:, :, 1:2]     
    
    #steering mse per step
    steering_mse_per_step = torch.mean((steering_pred - steering_true) ** 2, dim=[0, 2])
    steering_weighted_mse = torch.sum(steering_mse_per_step * seq_weights)
    
    #acceleration mse per step
    accel_mse_per_step = torch.mean((accel_pred - accel_true) ** 2, dim=[0, 2])
    accel_weighted_mse = torch.sum(accel_mse_per_step * seq_weights)
    
    #combined loss (equal weighting)
    combined_loss = steering_weighted_mse + accel_weighted_mse
    
    return combined_loss

#trying to see if combined magnitude weighted loss and exponential sequence weighted loss might provide some balance, but did not test out during training
def combined_weighted_mse_loss(y_pred, y_true, curve_factor=DEFAULT_CURVE_FACTOR, mag_factor=DEFAULT_CURVE_FACTOR, accel_factor=None):
    #if accel_factor not provided, use same factor for both outputs
    if accel_factor is None:
        accel_factor = mag_factor
    
    seq_len = y_pred.size(1)
    
    #sequence weights tensor
    seq_weights = torch.tensor([curve_factor ** (seq_len - i - 1) for i in range(seq_len)], 
                              device=y_pred.device)
    seq_weights = seq_weights / seq_weights.sum()
    
    #extract steering and acceleration components
    steering_pred = y_pred[:, :, 0:1]  
    accel_pred = y_pred[:, :, 1:2]     
    
    steering_true = y_true[:, :, 0:1]  
    accel_true = y_true[:, :, 1:2]     
    
    #steering loss with magnitude weighting
    steering_mag_weights = torch.exp(torch.abs(steering_true) * mag_factor)
    steering_squared_error = (steering_pred - steering_true) ** 2
    steering_mag_weighted_error = steering_mag_weights * steering_squared_error
    steering_loss = torch.sum(steering_mag_weighted_error) / torch.sum(steering_mag_weights)
    
    #acceleration loss with magnitude weighting
    accel_mag_weights = torch.exp(torch.abs(accel_true) * accel_factor)
    accel_squared_error = (accel_pred - accel_true) ** 2
    accel_mag_weighted_error = accel_mag_weights * accel_squared_error
    accel_loss = torch.sum(accel_mag_weighted_error) / torch.sum(accel_mag_weights)
    
    #apply sequence weighting
    combined_error = torch.cat([steering_squared_error, accel_squared_error], dim=2)
    seq_weighted_error = torch.mean(combined_error, dim=[0, 2]) * seq_weights
    seq_loss = torch.sum(seq_weighted_error)
    
    #normalise and combine losses
    final_weighted_error = steering_loss + accel_loss + seq_loss
    
    return final_weighted_error


#---------------------------
# Training helper functions
#---------------------------
def process_continuous_batch(model, video_batch, target_batch, speed_batch, file_ids, positions, 
                           hidden_manager, optimiser, loss_fnc, device,
                           clip_grad_norm=DEFAULT_CLIP_GRAD_NORM,
                           conv_grad_scaling=DEFAULT_CONV_GRAD_SCALING,
                           curve_factor=DEFAULT_CURVE_FACTOR,
                           mag_factor=DEFAULT_CURVE_FACTOR,
                           accel_factor=None):
    
    #move data to device
    video_batch = video_batch.to(device)
    target_batch = target_batch.to(device)
    if speed_batch is not None:
        speed_batch = speed_batch.to(device)
    
    #get appropriate hidden states based on file_ids and positions
    batch_size = len(file_ids)
    hidden_states = hidden_manager.get_hidden_states(file_ids, positions, batch_size)
    hidden_states = hidden_states.to(device)
    
    #training step - zero gradients
    optimiser.zero_grad()
    
    #forward pass with hidden states and speed data
    y_pred, new_hidden_states, _ = model(video_batch, speed_sequence=speed_batch, hidden_state=hidden_states)
    
    #compute specified loss with appropriate factors
    if loss_fnc == "sequence":
        weighted_loss = exponential_weighted_mse_loss(y_pred, target_batch, curve_factor, accel_factor)
    elif loss_fnc == "magnitude":
        weighted_loss = magnitude_weighted_mse_loss(y_pred, target_batch, mag_factor, accel_factor)
    elif loss_fnc == "combined":
        weighted_loss = combined_weighted_mse_loss(y_pred, target_batch, curve_factor, mag_factor, accel_factor)

    orig_loss = F.mse_loss(y_pred, target_batch)
    
    #backward pass
    weighted_loss.backward()
    
    #gradient scaling for convolutional head
    for name, param in model.named_parameters():
        if 'conv_head' in name and param.grad is not None:
            param.grad *= conv_grad_scaling
    
    #gradient clipping to prevent exploding gradients - AI suggestion
    if clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
    
    #update parameters
    optimiser.step()
    
    #apply parameter constraints if needed
    apply_param_constraints(model.ltc_cell)
    
    #update hidden states in manager with new values (detached to CPU)
    hidden_manager.update_hidden_states(file_ids, positions, new_hidden_states.detach())
    
    #compute mean absolute error
    with torch.no_grad():
        mae = torch.mean(torch.abs(y_pred - target_batch)) #combined mae but less interpertable
        steering_mae = torch.mean(torch.abs(y_pred[:, :, 0] - target_batch[:, :, 0])) #explicit steering mae
        accel_mae = torch.mean(torch.abs(y_pred[:, :, 1] - target_batch[:, :, 1])) #explicit acceleration mae
    
    #return batch results
    return {
        'loss': orig_loss.item(),
        'weighted_loss': weighted_loss.item(),
        'mae': mae.item(),
        'steering_mae': steering_mae.item(),
        'accel_mae': accel_mae.item(),
        'prediction': y_pred.detach().cpu(),
        'target': target_batch.detach().cpu()
    }

#for every epoch reset the hidden state in hidden state manager, loop over the batches and process each batch and obtain the batch metrics
#then calculate the average metrics over the total number of batches; utilises process_continuous_batch helper function
def train_continuous_model(model, train_loader, epochs, device, save_path, metrics_path=None, save_freq=1,
                          learning_rate=DEFAULT_LEARNING_RATE, 
                          conv_grad_scaling=DEFAULT_CONV_GRAD_SCALING,
                          curve_factor=DEFAULT_CURVE_FACTOR,
                          mag_factor=DEFAULT_CURVE_FACTOR,
                          accel_factor=None, #acceleration weighting factor
                          loss_type="magnitude", #magnitude by default
                          clip_grad_norm=DEFAULT_CLIP_GRAD_NORM,
                          telemetry_labels=None): #labels for metrics saving
    
    print(f"training with loss type: {loss_type}")
    if accel_factor is not None:
        print(f"using separate acceleration factor: {accel_factor}")
    else:
        print(f"using same factor for steering and acceleration")

    #setup training metrics - only track essential metrics
    history = {'weighted_loss': [], 
               'normal_loss': [], 
               'mae': [],
               'steering_mae': [],
               'accel_mae': []
               } 
    
    #setup optimiser
    optimiser = init_optimiser(model, learning_rate)
    
    #get model config for saving
    model_config = get_model_config(model)
    
    #initialise hidden state manager with model's hidden size
    hidden_manager = HiddenStateManager(
        hidden_size=model.ltc_cell.internal_neuron_size,
        device=device
    )
    
    #main epoch loop with tqdm
    epoch_bar = tqdm(range(epochs), desc="training epochs")
    
    for epoch in epoch_bar:
        epoch_loss = 0.0 #weighted loss
        epoch_normal_loss = 0.0
        epoch_mae = 0.0
        epoch_steering_mae = 0.0
        epoch_accel_mae = 0.0
        batch_count = 0
        
        #reset hidden states at the start of each epoch for consistent training
        hidden_manager.reset_hidden_state()
        
        #training phase
        model.train()
        
        #iterate through batches with expanded output from continuous loader
        for batch_data in train_loader:
            #unpack batch data - handle speed features if present
            if len(batch_data) == 5:  #with speed features
                video_batch, target_batch, speed_batch, file_ids, positions = batch_data

            else:  #without speed features (backwards compatibility)
                video_batch, target_batch, file_ids, positions = batch_data
                speed_batch = None
            
            #process batch with hidden state continuity
            batch_results = process_continuous_batch(
                model, video_batch, target_batch, speed_batch, file_ids, positions,
                hidden_manager, optimiser, loss_type, device, 
                clip_grad_norm, conv_grad_scaling, curve_factor, mag_factor, accel_factor
            )
            
            #update metrics
            epoch_loss += batch_results['weighted_loss']
            epoch_normal_loss += batch_results['loss']
            epoch_mae += batch_results['mae']
            epoch_steering_mae += batch_results['steering_mae']
            epoch_accel_mae += batch_results['accel_mae']
            batch_count += 1
            
            #update progress bar regularly
            epoch_bar.set_postfix(
                w_loss=f"{batch_results['weighted_loss']:.4f}",
                o_loss=f"{batch_results['loss']:.4f}",
                st_mae=f"{batch_results['steering_mae']:.4f}°",
                ac_mae=f"{batch_results['accel_mae']:.4f}",
                batches=f"{batch_count}"
            )

        
        #finalise epoch metrics
        avg_epoch_weighted_loss = epoch_loss / batch_count
        avg_epoch_normal_loss = epoch_normal_loss / batch_count 
        avg_epoch_mae = epoch_mae / batch_count
        avg_epoch_steering_mae = epoch_steering_mae / batch_count
        avg_epoch_accel_mae = epoch_accel_mae / batch_count
 
        history['weighted_loss'].append(avg_epoch_weighted_loss)
        history['normal_loss'].append(avg_epoch_normal_loss)
        history['mae'].append(avg_epoch_mae)  
        history['steering_mae'].append(avg_epoch_steering_mae)
        history['accel_mae'].append(avg_epoch_accel_mae)
        
        #update epoch bar with final metrics
        epoch_bar.set_postfix(
            avg_w_loss=f"{avg_epoch_weighted_loss:.4f}",
            avg_n_loss=f"{avg_epoch_normal_loss:.4f}",
            avg_epoch_mae=f"{avg_epoch_mae:.4f}",
        )
        
        #save model checkpoint and training metrics at the same frequency
        if save_path and (epoch + 1) % save_freq == 0:
            #save model checkpoint
            save_checkpoint(model, optimiser, save_path, epoch, 
                            {'weighted_loss': avg_epoch_weighted_loss,
                             'normal_loss': avg_epoch_normal_loss,
                             'mae': avg_epoch_mae,
                             'steering_mae': avg_epoch_steering_mae,
                             'accel_mae': avg_epoch_accel_mae}, 
                            model_config)
            
            #save training metrics if path is provided
            if metrics_path is not None:
                print(f"saving training metrics for epoch {epoch+1}")
                save_training_metrics(history, metrics_path, telemetry_labels)
    
    print("training completed")
    return history

#setup optimiser
def init_optimiser(model, learning_rate=DEFAULT_LEARNING_RATE):
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return optimiser

#apply parameter constraints after optimiser step, this is only used if the ltccell was initialised with argument "implicit constraint=True", 
#this applies neuron specific constraints taken from official github implementation
def apply_param_constraints(ltc_cell):
    #only needed when not using implicit constraints
    if not hasattr(ltc_cell, 'make_positive') or not isinstance(ltc_cell.make_positive, nn.Softplus):
        with torch.no_grad():
            #apply sensory weight constraints
            sensory_w = ltc_cell.params['sensory_w']
            mask = ltc_cell.params['sensory_sparsity_mask']
            sensory_w.data = torch.clamp(sensory_w.data * mask, min=0.001, max=100.0)
            
            #apply interneuron weight constraints  
            w = ltc_cell.params['w']
            mask = ltc_cell.params['sparsity_mask']
            w.data = torch.clamp(w.data * mask, min=0.001, max=100.0)
            
            #apply membrane capacitance constraints
            cm = ltc_cell.params['membrane_capacitance']
            cm.data = torch.clamp(cm.data, min=0.0001, max=1000.0)
            
            #apply leakage conductance constraints
            gleak = ltc_cell.params['leakage_conductance']
            gleak.data = torch.clamp(gleak.data, min=0.001, max=100.0)

#save checkpoint with model state and metadata
def save_checkpoint(model, optimiser, save_path, epoch, metrics, model_config):
    os.makedirs(save_path, exist_ok=True)
    
    checkpoint_path = os.path.join(save_path, f"model_epoch{epoch+1}.pt")

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'weighted_loss': metrics['weighted_loss'],
        'normal_loss': metrics.get('normal_loss', None),
        'mae': metrics.get('mae', None),  
        'steering_mae': metrics.get('steering_mae', None),
        'accel_mae': metrics.get('accel_mae', None),
        'model_config': model_config
    }, checkpoint_path)

#save after training for plotting
def save_training_metrics(history, save_path, telemetry_labels=None):
    #ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    #prepare metadata
    metadata = {
        'telemetry_labels': telemetry_labels if telemetry_labels is not None else ['output'],
        'num_epochs': len(history['weighted_loss']),
        'save_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    #save metadata
    with open(os.path.join(save_path, 'training_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    #save loss history - now with all metrics
    np.savez(os.path.join(save_path, 'training_loss.npz'), 
             epoch=np.arange(1, len(history['weighted_loss']) + 1),
             weighted_loss=np.array(history['weighted_loss']),
             normal_loss=np.array(history.get('normal_loss', [])),
             mae=np.array(history.get('mae', [])),
             steering_mae=np.array(history.get('steering_mae', [])),
             accel_mae=np.array(history.get('accel_mae', []))) 
    
    print(f"training metrics saved to {save_path}")

#extract model configuration for saving and used in loading function in model utilities
def get_model_config(model):
    return {
        'image_height': model.image_height,
        'image_width': model.image_width,
        'wiring_config': {
            'num_interneurons': model.wiring.num_interneurons,
            'num_command_neurons': model.wiring.num_command_neurons,
            'num_motor_neurons': model.wiring.num_motor_neurons,
            'sensory_fanout': model.wiring.sensory_fanout,
            'inter_fanout': model.wiring.inter_fanout,
            'recurrent_connections': model.wiring.recurrent_connections,
            'command_fanout': model.wiring.command_fanout,
            'seed': model.wiring.rndm_sd.get_state()[1][0] if hasattr(model.wiring.rndm_sd, 'get_state') else None
        },
        'num_filters': model.conv_head.num_filters,
        'features_per_filter': model.conv_head.features_per_filter,
        'input_mapping': model.ltc_cell.input_mapping,
        'output_mapping': model.ltc_cell.output_mapping,
        'ode_unfolds': model.ltc_cell.ode_unfolds,
        'implicit_constraints': model.ltc_cell.make_positive.__class__.__name__ == 'Softplus',
        'epsilon': model.ltc_cell.epsilon,
        'learning_rate': DEFAULT_LEARNING_RATE, 
        'conv_grad_scaling': DEFAULT_CONV_GRAD_SCALING,
        'curve_factor': DEFAULT_CURVE_FACTOR,
        'return_sequences': model.return_sequences
    }

