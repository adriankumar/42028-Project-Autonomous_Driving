import torch
import os, json
import numpy as np
import matplotlib.pyplot as plt
from LTC.model.neural_wiring import NeuralCircuitPolicy as ncp
from LTC.model.convLTC import ConvLTC
import torch.nn.functional as F
import cv2

#load convLTC model
def load_model(checkpoint_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    #extract model config
    model_config = checkpoint.get('model_config', None)
    
    if model_config is None:
        raise ValueError("Checkpoint does not contain model configuration")
    
    #extract wiring config
    wiring_config = model_config['wiring_config']
    
    #recreate wiring
    wire = ncp(
        inter_neurons=wiring_config['num_interneurons'],
        command_neurons=wiring_config['num_command_neurons'],
        motor_neurons=wiring_config['num_motor_neurons'],
        outgoing_sensory_neurons=wiring_config['sensory_fanout'],
        outgoing_inter_neurons=wiring_config['inter_fanout'],
        num_of_recurrent_connections=wiring_config['recurrent_connections'],
        outgoing_command_neurons=wiring_config['command_fanout'],
        seed=wiring_config.get('seed', None)
    )
    
    #build wiring with correct input dimension
    sensory_size = model_config['num_filters'] * model_config['features_per_filter'] + 8 #additional 8 comes from the speed feature embedding where the final linear layer has output dim 16 -> 8
    wire.build(sensory_size)
    
    #create model
    model = ConvLTC(
        wiring=wire,
        image_height=model_config['image_height'],
        image_width=model_config['image_width'],
        num_filters=model_config['num_filters'],
        features_per_filter=model_config['features_per_filter'],
        return_sequences=model_config['return_sequences'],
        input_mapping=model_config['input_mapping'],
        output_mapping=model_config['output_mapping'],
        ode_unfolds=model_config['ode_unfolds'],
        epsilon=model_config.get('epsilon', 1e-8),
        implicit_constraints=model_config.get('implicit_constraints', False)
    )
    
    #load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    #move model to device
    model = model.to(device)
    
    print(f"Model loaded from {checkpoint_path} (epoch {checkpoint['epoch']})")
    non_zero_params = sum(param.count_nonzero() for param in model.parameters())
    total_params = sum(param.numel() for param in model.parameters())
    print(f"model has {non_zero_params}/{total_params} non-zero parameters")
    
    return model

#plot loss data saved during training
def plot_loss(metrics_path, loss_file='training_loss.npz'):
    #load loss data
    loss_data = np.load(os.path.join(metrics_path, loss_file))
    
    #create figure with 2 rows (one for losses, one for MAE)
    plt.figure(figsize=(12, 10))
    
    #plot losses
    plt.subplot(2, 1, 1)
    plt.plot(loss_data['epoch'], loss_data['weighted_loss'], 'r-', label='weighted loss') #weighted magnitude loss
    
    plt.plot(loss_data['epoch'], loss_data['normal_loss'], 'b-', label='normal loss') #standard mse loss, but this was not used to update model parameters, just to see what it looks like in general for further evaluation
    
    plt.title('training losses')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    
    #plot mae metrics
    plt.subplot(2, 1, 2)
    
    #plot overall mae
    plt.plot(loss_data['epoch'], loss_data['mae'], 'g-', label='overall mae') #which combined steering and accel mae, although more enigmatic to interpret, keep it anyways cos we might see something interesting
    
    #plot steering mae 
    plt.plot(loss_data['epoch'], loss_data['steering_mae'], 'c-', label='steering mae')
    
    #plot acceleration mae 
    plt.plot(loss_data['epoch'], loss_data['accel_mae'], 'm-', label='acceleration mae')
    
    plt.title('mean absolute error metrics')
    plt.xlabel('epoch')
    plt.ylabel('mae')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

#plot weighted loss, steering and car accel mae seperately for closer inspection
def plot_detailed_metrics(metrics_path, loss_file='training_loss.npz', metadata_file='training_metadata.json'):
    #load loss data and metadata
    loss_data = np.load(os.path.join(metrics_path, loss_file))
    
    try:
        with open(os.path.join(metrics_path, metadata_file), 'r') as f:
            metadata = json.load(f)
            
        telemetry_labels = metadata.get('telemetry_labels', ['output'])
        num_epochs = metadata.get('num_epochs', len(loss_data['epoch']))
        
        print(f"training session info:")
        print(f"output labels: {telemetry_labels}")
        print(f"total epochs: {num_epochs}")
        
        if 'save_time' in metadata:
            print(f"saved on: {metadata['save_time']}")

    except (FileNotFoundError, json.JSONDecodeError):
        print(f"metadata file not found or invalid: {metadata_file}")
    
    #create multi-plot figure
    num_plots = 3 #weight loss + steering + accel
    plt.figure(figsize=(12, 4 * num_plots))
    
    #plot weighted loss
    plt.subplot(num_plots, 1, 1)
    plt.plot(loss_data['epoch'], loss_data['weighted_loss'], 'r-', linewidth=2)
    plt.title('weighted training loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    
    #plot steering metrics
    plt.subplot(num_plots, 1, 2)
    plt.plot(loss_data['epoch'], loss_data['steering_mae'], 'b-', linewidth=2)   
    plt.title('steering angle mae')
    plt.xlabel('epoch')
    plt.ylabel('degrees')
    plt.grid(True)
        
    #plot acceleration metrics
    plt.subplot(num_plots, 1, 3)
    plt.plot(loss_data['epoch'], loss_data['accel_mae'], 'g-', linewidth=2)      
    plt.title('car acceleration mae')
    plt.xlabel('epoch')
    plt.ylabel('acceleration units')
    plt.grid(True)
        
    
    plt.tight_layout()
    plt.show()

#expects predictions input to be a list with elements whos shape is 1 x 1 (seq_len x output_dim) as a torch tensor
#but can also plot a specific prediction using the idx arugment (steering angle, car acceleration)
#used for inference validation
def plot_predictions(predictions, true_values, output_idx=None, output_name="prediction"):
    #convert predictions to numpy array
    pred_tensor = torch.stack(predictions)
    
    #define output names based on input type
    output_names = []
    if isinstance(true_values, dict):
        output_names = list(true_values.keys())
    else:
        output_names = ["steering_angle", "car_accel"] if pred_tensor.shape[-1] == 2 else [f"output_{i}" for i in range(pred_tensor.shape[-1])]
    
    #check if we're plotting a specific output or all
    if output_idx is not None:
        #extract specific output channel
        if len(pred_tensor.shape) == 3:
            #format: [frames, 1, n_outputs]
            pred_array = pred_tensor[:, 0, output_idx].cpu().numpy()
        elif len(pred_tensor.shape) == 2:
            #format: [frames, n_outputs]
            pred_array = pred_tensor[:, output_idx].cpu().numpy()
        elif len(pred_tensor.shape) == 1:
            #simplest case: [frames] (single output model)
            pred_array = pred_tensor.cpu().numpy()
        else:
            raise ValueError(f"unexpected prediction tensor shape: {pred_tensor.shape}")
        
        #get true values for the specific output
        if isinstance(true_values, dict):
            #handle dictionary input (from telemetry)
            current_output_name = output_names[output_idx] if output_idx < len(output_names) else output_name
            true_array = true_values[current_output_name]
        elif isinstance(true_values, np.ndarray) and true_values.ndim > 1:
            #handle numpy array with multiple outputs
            true_array = true_values[:, output_idx]
        else:
            #single output array
            true_array = true_values
            
        #ensure length matches ground truth
        min_len = min(len(pred_array), len(true_array))
        pred_array = pred_array[:min_len]
        true_array = true_array[:min_len]

        #create simple plot
        plt.figure(figsize=(10, 6))
        plt.plot(pred_array, 'b-', label=f'predicted {output_name}')
        plt.plot(true_array, 'r-', label=f'ground truth {output_name}')
        plt.title(f'{output_name} prediction')
        plt.xlabel('frame')
        plt.ylabel('value')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        #calculate and print error metrics
        mae = np.mean(np.abs(pred_array - true_array))
        mse = np.mean((pred_array - true_array)**2)
        print(f"{output_name} mae: {mae:.4f}, mse: {mse:.4f}")
        
    else:
        #plot all outputs
        n_outputs = pred_tensor.shape[-1]
        
        plt.figure(figsize=(12, 4 * n_outputs))
        
        for i in range(n_outputs):
            if len(pred_tensor.shape) == 3:
                pred_array = pred_tensor[:, 0, i].cpu().numpy()
            elif len(pred_tensor.shape) == 2:
                pred_array = pred_tensor[:, i].cpu().numpy()
            
            current_output_name = output_names[i] if i < len(output_names) else f"output_{i}"
            
            #get true values for this output
            if isinstance(true_values, dict):
                #handle dictionary input
                true_array = true_values[current_output_name]
            elif isinstance(true_values, np.ndarray) and true_values.ndim > 1:
                #handle numpy array with multiple outputs
                true_array = true_values[:, i]
            else:
                #single output array
                true_array = true_values
            
            #ensure length matches
            min_len = min(len(pred_array), len(true_array))
            pred_array = pred_array[:min_len]
            true_array = true_array[:min_len]
            
            plt.subplot(n_outputs, 1, i+1)
            plt.plot(pred_array, 'b-', label=f'predicted {current_output_name}')
            plt.plot(true_array, 'r-', label=f'ground truth {current_output_name}')
            plt.title(f'{current_output_name} prediction')
            plt.xlabel('frame')
            plt.ylabel('value')
            plt.legend()
            plt.grid(True)
            
            #calculate and print error metrics
            mae = np.mean(np.abs(pred_array - true_array))
            mse = np.mean((pred_array - true_array)**2)
            print(f"{current_output_name} mae: {mae:.4f}, mse: {mse:.4f}")
        
        plt.tight_layout()
        plt.show()

#plot the hidden state activity over time
def plot_hidden_state_activity(hidden_states, title="hidden state activity", max_neurons=20):
    #convert list of tensors to numpy array
    if isinstance(hidden_states[0], torch.Tensor):
        hidden_array = torch.stack(hidden_states).cpu().numpy()
    else:
        hidden_array = np.array(hidden_states)
    
    #determine how many neurons to plot
    n_neurons = hidden_array.shape[1]
    plot_neurons = min(n_neurons, max_neurons)
    
    plt.figure(figsize=(12, 8))
    
    #plot each neuron's activity
    for i in range(plot_neurons):
        plt.plot(hidden_array[:, i], label=f"neuron {i+1}" if i < 10 else None)
    
    if plot_neurons < n_neurons:
        plt.title(f"{title} (showing {plot_neurons} of {n_neurons} neurons)")
    else:
        plt.title(title)
    
    plt.xlabel("frame")
    plt.ylabel("activation")
    
    if plot_neurons <= 10:
        plt.legend()
    
    plt.grid(True)
    plt.show()
    
    #also plot overall activity statistics
    plt.figure(figsize=(12, 4))
    
    #calculate statistics
    mean_activity = np.mean(hidden_array, axis=1)
    max_activity = np.max(hidden_array, axis=1)
    min_activity = np.min(hidden_array, axis=1)
    
    plt.plot(mean_activity, 'b-', label='mean')
    plt.plot(max_activity, 'r-', label='max')
    plt.plot(min_activity, 'g-', label='min')
    
    plt.title("hidden state activity statistics")
    plt.xlabel("frame")
    plt.ylabel("activation")
    plt.legend()
    plt.grid(True)
    plt.show()


#visualbackprop implementation based on https://arxiv.org/pdf/1611.05418
def visualbackprop(model, input_tensor, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #ensure model is in evaluation mode
    model.eval()
    
    #move input to device if needed
    input_tensor = input_tensor.to(device)
    
    #expect input shape [batch, seq_len, height, width, channels]
    batch_size, seq_len = input_tensor.shape[0], input_tensor.shape[1]
    
    with torch.no_grad():
        #collect feature maps from convolutional layers
        activation_maps = []
        
        #reshape for convolutional processing
        x = input_tensor.reshape(-1, input_tensor.shape[-3], input_tensor.shape[-2], input_tensor.shape[-1])
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        
        #normalise like in the conv_head
        x = (x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2, 3), keepdim=True) + 1e-5)
        
        #store input as first activation
        activation_maps.append(x)
        
        #process through conv layers
        for conv in model.conv_head.conv_layers:
            x = F.relu(conv(x))
            activation_maps.append(x)
        
        #average each feature map along channel dimension
        averaged_maps = []
        for feat_map in activation_maps:
            avg_map = torch.mean(feat_map, dim=1, keepdim=True)
            averaged_maps.append(avg_map)
        
        #backpropagate visualisation using visual backprop algorithm
        #start with the deepest feature map
        visualisation = averaged_maps[-1]
        
        #work backwards through the layers
        for i in range(len(averaged_maps) - 2, -1, -1):
            #scale up to match previous layer's size
            target_size = averaged_maps[i].shape[2:]
            scaled_up = F.interpolate(visualisation, size=target_size, mode='bilinear', align_corners=False)
            
            #point-wise multiplication
            visualisation = scaled_up * averaged_maps[i]
        
        #reshape to match input structure
        vis_height, vis_width = visualisation.shape[2], visualisation.shape[3]
        
        #convert to numpy for final processing
        visualisation = visualisation.reshape(batch_size, seq_len, vis_height, vis_width).cpu().numpy()
        
        #normalize visualisation maps
        for b in range(batch_size):
            for s in range(seq_len):
                vis = visualisation[b, s]
                if vis.max() > vis.min():
                    visualisation[b, s] = (vis - vis.min()) / (vis.max() - vis.min())
        
        return visualisation
    
#plot the saliency maps
def plot_saliency_maps(image_batch, saliency_maps, alpha=1, n_samples=None, save_path=None):
    #extract dimensions
    batch_size = min(saliency_maps.shape[0], image_batch.shape[0])
    seq_len = min(saliency_maps.shape[1], image_batch.shape[1])
    
    #limiting samples for display
    if n_samples is None:
        n_samples = min(batch_size, 4)
    
    fig, axes = plt.subplots(n_samples, seq_len, figsize=(seq_len * 3, n_samples * 3))
    
    #handle case when only one sample or one frame
    if n_samples == 1 and seq_len == 1:
        axes = np.array([[axes]])
    elif n_samples == 1:
        axes = axes.reshape(1, -1)
    elif seq_len == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_samples):
        for j in range(seq_len):
            #get original image and convert to numpy
            img = image_batch[i, j].cpu().numpy()
            
            #normalise image
            if img.max() > 1.0:
                img = img / 255.0
            
            #get corresponding saliency map
            saliency = saliency_maps[i, j]
            
            #ensure saliency matches image dimensions
            if saliency.shape != img.shape[:2]:
                saliency = cv2.resize(saliency, (img.shape[1], img.shape[0]))
            
            #create red-black colormap directly
            #normalise saliency to 0-1 range
            saliency_norm = saliency / saliency.max() if saliency.max() > 0 else saliency
            
            #create RGB map with red channel only
            mask_rgb = np.zeros((saliency.shape[0], saliency.shape[1], 3), dtype=np.float32)
            mask_rgb[:,:,0] = saliency_norm  #red channel
            
            #add some pink/purple tint by adding a little blue
            mask_rgb[:,:,2] = saliency_norm * 0.5  #blue channel at half intensity for purplish tint
            
            #blend images
            alpha = alpha #set to 1 for no image blend, decrease for image blend
            overlay = img * (1-alpha) + mask_rgb * alpha
            
            axes[i, j].imshow(overlay)
            axes[i, j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()