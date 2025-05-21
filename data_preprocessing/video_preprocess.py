import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import albumentations as A
from tqdm import tqdm


#constants
# train_path = r"datasets\Comma_ai_dataset\train"
# val_path = r"datasets\Comma_ai_dataset\val"
# test_path = r"datasets\Comma_ai_dataset\test"

train_files = ['2016-06-08--11-46-01.h5', #2.63GB <-- smallest
               '2016-02-08--14-56-28.h5', #3.80GB
               '2016-06-02--21-39-29.h5', #6.44GB   ..to
               '2016-02-02--10-16-58.h5', #8.05GB
               '2016-03-29--10-50-20.h5', #11.2GB
               '2016-02-11--21-32-47.h5'] #12.3GB <-- largest
               
val_files = ['2016-04-21--14-48-08.h5', #4.39GB <-- smallest
             '2016-01-30--11-24-51.h5'] #7.61GB <-- largest

test_files = ['2016-01-31--19-19-25.h5', #2.92GB <-- smallest
              '2016-05-12--22-20-00.h5', #7.47GB        ..to
              '2016-01-30--13-46-00.h5'] #8.49GB <-- largest

telemetry_keys = ['steering_angle', 'steering_torque', 'speed', 'speed_abs', 'car_accel', 'brake', 'brake_user', 'brake_computer', 'standstill', 'times']


#this helper aligns telemetry data to video frames
def align_telemetry(data, ptr):
    ptr_int = ptr.astype(int)
    max_frame = ptr_int.max() #returns max index value in the ptr list which is the final frame index

    aligned = np.empty(max_frame + 1, dtype=data.dtype) #create emtpy numpy array with length of frames

    for frame in range(max_frame + 1):
        idx = np.where(ptr_int == frame)[0] #idx creates a list of telemetry data that belongs to the frame index from cam_1_ptr

        #if idx is not empty then use the last telemetry data
        if idx.size > 0:
            aligned[frame] = data[idx[-1]] #use last index as aligned value

        #else its missing, using the previous value to ensure temporal continuity
        else:
            aligned[frame] = aligned[frame - 1] #use previous index

    return aligned

#this plots selected telemetry series over frames (numpy array input)
def plot_telemetry(telemetry_data, labels, start=0, end=-1):

    for label in labels:
        if label in telemetry_data:
            plt.plot(telemetry_data[label][start:end], label=label)
    plt.legend()
    plt.xlabel('frame')
    plt.ylabel('value')
    plt.show()

#memory efficient version that augments individual clips instead of entire videos
def augment_clip(clip, mode=''):
    #this adds noise to every frame
    if mode == 'noise': 
        aug = A.GaussNoise(std_range=(0.3, 0.3), per_channel=True, noise_scale_factor=1, p=1.0)
        out = np.empty_like(clip)

        for i, frame in enumerate(clip):
            #albumentations expects H x W x C uint8
            img = frame.transpose(1, 2, 0)
            img = aug(image=img)['image']
            out[i] = img.transpose(2, 0, 1)

        return out
    
    #this dims or brightens all frames in the clip
    if mode in ('dim', 'light'):
        #brightness_limit negative dims, positive brightens
        limit = (-0.3, -0.1) if mode == 'dim' else (0.1, 0.3)
        aug = A.RandomBrightnessContrast(brightness_limit=limit,
                                         contrast_limit=0,
                                         p=1.0)
        out = clip.copy()
        for i in range(len(clip)):
            img = clip[i].transpose(1, 2, 0)
            img = aug(image=img)['image']
            out[i] = img.transpose(2, 0, 1)

        return out
    
    #this returns original if no valid augmentation mode
    return clip

#function to use with efficient loading of segments, used during inference
def load_sample(datasplit_path, file_name, telemetry_labels, augment='', normalise=True, start=0, end=-1, extract_speed_feature=False):
    #hardcoded file locations
    camera_file = f"{datasplit_path}\camera\{file_name}" 
    telemetry_file = f"{datasplit_path}\labels\{file_name}"

    if not os.path.isfile(camera_file) or not os.path.isfile(telemetry_file): 
        raise ValueError(f"Either {camera_file} or {telemetry_file} is invalid")

    #load only the segmented sequence in memory during inference
    with h5py.File(camera_file, 'r') as f:
        total_frames = f['X'].shape[0]
        if end == -1:
            end = total_frames
        
        #loading segment in memory
        video_data = np.asarray(f['X'][start:end]) 
    
    #apply augmentation if requested
    if augment in ('noise', 'dim', 'light'):
        video_data = augment_clip(video_data, augment)   

    if normalise:
        video_data = video_data.astype(np.float32) / 255.0 #normalise pixel values

    #load telemetry and slice directly
    telemetry_data = {}
    speed_feature = None
    
    with h5py.File(telemetry_file, 'r') as f:
        raw_ptr = np.asarray(f['cam1_ptr'])
        
        #determine actual indices for telemetry slicing
        for label in telemetry_labels:
            if label not in f:
                print(f"label not in telemetry... skipping")
                continue

            raw = np.asarray(f[label]) 
            
            #add scaling for specific labels if known
            if label == 'steering_angle':
                raw = raw / 10 #offical comma ai dataset scales steering angle by 10, divide by 10 to get real steering angles
            
            aligned = align_telemetry(raw, raw_ptr)
            #slice the aligned telemetry directly
            telemetry_data[label] = aligned[start:end]
        
        #extract speed feature if requested (added as modification to accomodate for car accel predictions, original version didnt use telemetry data as features)
        if extract_speed_feature and 'speed_abs' in f:
            raw_speed = np.asarray(f['speed_abs'])
            aligned_speed = align_telemetry(raw_speed, raw_ptr)
            speed_feature = aligned_speed[start:end].astype(np.float32)
            #reshape to match expected format [frames, 1]
            speed_feature = np.expand_dims(speed_feature, axis=-1)

    return (video_data, telemetry_data, speed_feature) if extract_speed_feature else (video_data, telemetry_data) #all returned as numpy arrays

#process a video segment frame by frame or seq_len by seq_len, maintaining hidden state continuity
def process_video_inference(model, video_data, device, speed_data=None, seq_len=1, normalise=True):
    model.eval()
    
    f, c, h, w = video_data.shape
    predictions = []
    hidden_states = []
    hidden_state = None
    
    #ensure model is in evaluation mode
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, f, seq_len), desc="processing frames"):
            #get current segment
            end_idx = min(i + seq_len, f)
            seg_len = end_idx - i
            
            #in the case that the remaining sequence length in the video segment is smaller than the actual defined sequence length, i.e seq_len = 14, remaining sequence in last segment has length 4
            #then we simply create a new frame segment in the same numpy shape, then append the remaining sequence length in the data to that
            if seg_len < seq_len:
                frame_segment = np.zeros((seq_len, c, h, w), dtype=video_data.dtype) #create new empty numpy array in same video format shape
                frame_segment[:seg_len] = video_data[i:end_idx] #add remaining video data to the segment
                
                #same thing if using speed data
                if speed_data is not None:
                    speed_segment = np.zeros((seq_len, speed_data.shape[1]), dtype=speed_data.dtype)
                    speed_segment[:seg_len] = speed_data[i:end_idx]

            #else segment the video and speed data as normal        
            else:
                frame_segment = video_data[i:end_idx]
                speed_segment = speed_data[i:end_idx] if speed_data is not None else None
            
            #prepare tensor
            if normalise and video_data.dtype == np.uint8:
                frame_segment = frame_segment.astype(np.float32) / 255.0
                
            tensor_frames = torch.tensor(frame_segment, dtype=torch.float32, device=device)
            #reshape to [1, seq_len, H, W, C] - batch major format for conv input which internally handles the time major format reshaping
            tensor_frames = tensor_frames.permute(0, 2, 3, 1).unsqueeze(0)
            
            #prepare speed data if provided
            if speed_segment is not None:
                tensor_speed = torch.tensor(speed_segment, dtype=torch.float32, device=device)
                #reshape to [1, seq_len, 1] batch major format for speed embedding input
                tensor_speed = tensor_speed.unsqueeze(0)
            else:
                tensor_speed = None
            
            #run inference
            prediction, hidden_state, all_hidden_states = model(
                input_sequence=tensor_frames, #visual features
                speed_sequence=tensor_speed, #speed feature
                hidden_state=hidden_state
            )
            
            #store results
            for j in range(seg_len):
                if model.return_sequences:
                    pred = prediction[0, j]  #[batch, seq, features] -> [features]
                    h_state = all_hidden_states[0, j]  #[batch, seq, hidden] -> [hidden]
                else:
                    #if not returning sequences, only the last frame has a prediction
                    if j == seg_len - 1:
                        pred = prediction[0]  #[batch, features] -> [features]
                        h_state = hidden_state[0]  #[batch, hidden] -> [hidden]
                    else:
                        continue
                
                predictions.append(pred)
                hidden_states.append(h_state)
    
    return predictions, hidden_states