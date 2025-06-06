import torch
import numpy as np
from data_preprocessing.model_utilities import load_model
import cv2

#hardcoded path to model
MODEL_PATH = r"LTC\checkpoint_weights\model_epoch170.pt"
#sequence length for model input
SEQ_LEN = 1  #change this to experiment with different sequence lengths
#device for model inference
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#global variables
model = None
frame_buffer = []
hidden_state = None

#helper function that loads the model in and sets return_sequence variable to false so that we can use its most recent prediction
#we should really rename the function
def load_steering_model():
    global model
    if model is None:
        try:
            model = load_model(MODEL_PATH, device=DEVICE)
            #if SEQ_LEN = 1, we want only the last prediction
            if SEQ_LEN == 1:
                model.return_sequences = False
            print(f"model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"error loading model: {str(e)}")
    return model

#function that takes the currently displayed frame from the video display, and prepares it into the correct datatype and shape
#for model prediction, returning the video feature and speed feature as a tensor
def prepare_input_frame(frame, speed_value, normalise=True):
    global frame_buffer
    
    #copy frame to avoid modifying the original
    frame_copy = frame.copy()
    
    #convert frame to float32 if it's not already
    if frame_copy.dtype != np.float32:
        frame_copy = frame_copy.astype(np.float32)
    
    #normalise if requested (0-255 to 0-1)
    if normalise:
        frame_copy = frame_copy / 255.0
    
    #add to buffer
    frame_buffer.append(frame_copy)
    
    #if buffer is not full, return None
    if len(frame_buffer) < SEQ_LEN:
        return None
    
    #keep only the most recent SEQ_LEN frames
    if len(frame_buffer) > SEQ_LEN:
        frame_buffer = frame_buffer[-SEQ_LEN:]
    
    #stack frames along sequence dimension
    input_array = np.stack(frame_buffer, axis=0) #shape: (seq_len, h, w, c)
    
    #convert to tensor
    input_tensor = torch.tensor(input_array, dtype=torch.float32, device=DEVICE)
    
    #add batch dimension
    input_tensor = input_tensor.unsqueeze(0) #shape: (1, seq_len, h, w, c)
    
    #prepare speed tensor with same sequence length
    speed_array = np.full((SEQ_LEN, 1), speed_value, dtype=np.float32)
    speed_tensor = torch.tensor(speed_array, dtype=torch.float32, device=DEVICE)
    speed_tensor = speed_tensor.unsqueeze(0) #shape: (1, seq_len, 1)
    
    return input_tensor, speed_tensor

#calls prepare input frame and performs model inference, returns the clamped prediction values for steering and acceleration
def predict_steering_and_acceleration(frame, speed_value, normalise=True, clamp_steering=(-720, 720), clamp_accel=(-5.0, 5.0)):
    global model, hidden_state
    
    if model is None:
        load_steering_model()
        #if model still not loaded, return None
        if model is None:
            return None, None
    
    #prepare input with speed
    input_tensors = prepare_input_frame(frame, speed_value, normalise)
    
    #if buffer not full, return None
    if input_tensors is None:
        return None, None
    
    input_tensor, speed_tensor = input_tensors #unpack
    
    #perform inference
    try:
        with torch.no_grad():
            predictions, hidden_state, _ = model(input_tensor, speed_tensor, hidden_state)
        
        #extract both predictions
        if model.return_sequences:
            steering_pred = predictions[0, -1, 0].item()  #last sequence, first output
            accel_pred = predictions[0, -1, 1].item() #last sequence, second output
        else:
            steering_pred = predictions[0, 0].item() #first output
            accel_pred = predictions[0, 1].item() #second output
        
        #clamp to ranges
        steering_pred = max(clamp_steering[0], min(clamp_steering[1], steering_pred))
        accel_pred = max(clamp_accel[0], min(clamp_accel[1], accel_pred))
        
        return steering_pred, accel_pred
    
    except Exception as e:
        print(f"error during inference: {str(e)}")
        return None, None

#reset model
def reset_model_state():
    global hidden_state, frame_buffer
    hidden_state = None
    frame_buffer = []

#generate saliency map using visualbackprop
#note we remake the function here instead of calling it from video preprocess to handle gui specific cases
def generate_saliency_map(frame, normalise=True):
    global model, hidden_state
    
    if model is None:
        load_steering_model()
        #if model still not loaded, return None
        if model is None:
            return None
    
    #ensure model is in evaluation mode
    model.eval()
    
    #prepare input just like for prediction
    frame_copy = frame.copy()
    
    #convert to float32
    if frame_copy.dtype != np.float32:
        frame_copy = frame_copy.astype(np.float32)
    
    #normalise if requested
    if normalise:
        frame_copy = frame_copy / 255.0
    
    #create a tensor
    input_tensor = torch.tensor(frame_copy, dtype=torch.float32, device=DEVICE)
    
    #add batch and sequence dimensions [1, 1, H, W, C]
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    
    #process saliency using visualbackprop
    with torch.no_grad():
        #collect feature maps from convolutional layers
        activation_maps = []
        
        #reshape for convolutional processing
        x = input_tensor.reshape(-1, input_tensor.shape[-3], input_tensor.shape[-2], input_tensor.shape[-1])
        x = x.permute(0, 3, 1, 2)  #NHWC -> NCHW
        
        #normalize like in the conv_head
        x = (x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2, 3), keepdim=True) + 1e-5)
        
        #store input as first activation
        activation_maps.append(x)
        
        #process through conv layers
        for conv in model.conv_head.conv_layers:
            x = torch.nn.functional.relu(conv(x))
            activation_maps.append(x)
        
        #average each feature map along channel dimension
        averaged_maps = []
        for feat_map in activation_maps:
            avg_map = torch.mean(feat_map, dim=1, keepdim=True)
            averaged_maps.append(avg_map)
        
        #backpropagate visualisation using visual backprop algorithm
        visualisation = averaged_maps[-1]
        
        #work backwards through the layers
        for i in range(len(averaged_maps) - 2, -1, -1):
            #scale up to match previous layer's size
            target_size = averaged_maps[i].shape[2:]
            scaled_up = torch.nn.functional.interpolate(
                visualisation, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
            
            #point-wise multiplication
            visualisation = scaled_up * averaged_maps[i]
        
        #reshape back to original dimensions
        vis_height, vis_width = visualisation.shape[2], visualisation.shape[3]
        visualisation = visualisation.reshape(1, 1, vis_height, vis_width).cpu().numpy()
        
        #extract and normalise the single saliency map
        saliency = visualisation[0, 0]
        if saliency.max() > saliency.min():
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        
        #create red-black colourmap with purple tint (matching model_utilities.py)
        saliency_norm = saliency / saliency.max() if saliency.max() > 0 else saliency
        
        #create RGB map with red channel only
        mask_rgb = np.zeros((saliency.shape[0], saliency.shape[1], 3), dtype=np.float32)
        mask_rgb[:,:,0] = saliency_norm  #red channel
        
        #add some pink/purple tint by adding a little blue
        mask_rgb[:,:,2] = saliency_norm * 0.5  #blue channel at half intensity for purplish tint
        
        #convert to 0-255 range
        heatmap = (mask_rgb * 255).astype(np.uint8)
        
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        
        return heatmap
    
#enable synaptic weight capture for visualisation
def enable_synaptic_weight_capture():
    if model is not None:
        model.ltc_cell.set_synaptic_weight_capture(True)

#disable synaptic weight capture for visualisation
def disable_synaptic_weight_capture():
    if model is not None:
        model.ltc_cell.set_synaptic_weight_capture(False)

#get the last captured synaptic weights from the model
def get_synaptic_weights():
    if model is not None:
        return model.ltc_cell.get_synaptic_weights()
    return None