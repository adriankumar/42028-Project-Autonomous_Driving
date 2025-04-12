from data_reading import comma_ai_data, bdd100k_data
import h5py
import numpy as np
import matplotlib.pyplot as plt
import math
import os


#SPECIFY HERE WHAT FILES FROM COMMA AI YOU WANT TO INCLUDE IN TRAINING, VALIDATION AND TESTING
#========== hardcoded list specifiying file name ==========
#add more files names to the list if needed or individual if dealing with larger files
TRAINING_FILES = ['2016-06-08--11-46-01']
VALIDATION_FILES = [] #'2016-04-21--14-48-08', leaving out for now because its 4GB
TESTING_FILES = ['2016-01-31--19-19-25']

def preprocess_comma_ai_data(comma_ai_data):
    #create output dictionary with the same structure
    processed_data = {
        'train': [],
        'val': [],
        'test': []
    }
    
    #process files for each split
    for file_name in TRAINING_FILES:
        if process_single_file(comma_ai_data, 'train', file_name, processed_data):
            print(f"Successfully processed training file: {file_name}")
    
    for file_name in VALIDATION_FILES:
        if process_single_file(comma_ai_data, 'val', file_name, processed_data):
            print(f"Successfully processed validation file: {file_name}")
    
    for file_name in TESTING_FILES:
        if process_single_file(comma_ai_data, 'test', file_name, processed_data):
            print(f"Successfully processed testing file: {file_name}")
    
    return processed_data

def process_single_file(comma_ai_data, split, file_name, processed_data):
    #find camera and label files for the specified file_name
    camera_file = None
    label_file = None
    
    #search through the camera files to find the matching one
    for path in comma_ai_data['camera'][split]:
        if file_name in path:
            camera_file = path
            break
    
    #search through the label files to find the matching one
    for path in comma_ai_data['labels'][split]:
        if file_name in path:
            label_file = path
            break
    
    #check if both files were found
    if camera_file is None or label_file is None:
        print(f"Could not find camera or label file for {file_name} in {split} split")
        return False
    
    #process the matched files
    try:
        #load camera data from h5 file
        with h5py.File(camera_file, 'r') as cam_h5:
            #extract frames from h5 file, shape is frames x channels x height x width which is what pytorch expects
            frames = np.array(cam_h5['X'])
            
            #normalise frames to [0-1] range
            frames = frames.astype(np.float32) / 255.0
        
        #load label data from h5 file
        with h5py.File(label_file, 'r') as label_h5:
            #extract all 13 telemetry features
            telemetry_features = {
                'blinker': np.array(label_h5['blinker']),
                'brake_computer': np.array(label_h5['brake_computer']),
                'brake_user': np.array(label_h5['brake_user']),
                'brake': np.array(label_h5['brake']),  
                'cam1_ptr': np.array(label_h5['cam1_ptr']),
                'car_accel': np.array(label_h5['car_accel']),
                'gas': np.array(label_h5['gas']),
                'speed': np.array(label_h5['speed']),
                'speed_abs': np.array(label_h5['speed_abs']),
                'standstill': np.array(label_h5['standstill']),
                'steering_angle': np.array(label_h5['steering_angle']) / 10.0, #in official github from comma ai, when using steering angle they divide by 10 indicating a scaling factor in data collection, so we do the same 
                'steering_torque': np.array(label_h5['steering_torque']),
                'times': np.array(label_h5['times'])
            }
            
            #get cam1_ptr for alignment
            cam1_ptr = telemetry_features['cam1_ptr']
            
            #align telemetry data with camera frames using cam1_ptr
            aligned_telemetry = align_telemetry_to_frames(telemetry_features, cam1_ptr, frames.shape[0])
            
            #store the processed data
            sequence_data = {
                'frames': frames,
                'telemetry': aligned_telemetry,
                'file_name': file_name
            }
            
            #append to the appropriate split
            processed_data[split].append(sequence_data)
            
            return True
    
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return False

def align_telemetry_to_frames(telemetry_features, cam1_ptr, num_frames):
    #create an empty array for aligned telemetry with shape (num_frames, 13)
    aligned_telemetry = np.zeros((num_frames, 13), dtype=np.float32)
    
    #for each frame, find the last log entry pointing to it
    for frame_idx in range(num_frames):
        #find all log entries pointing to this frame
        log_indices = np.where(cam1_ptr == frame_idx)[0]
        
        if len(log_indices) > 0:
            #get the last log entry pointing to this frame
            last_log_idx = log_indices[-1]
            
            #extract telemetry values for this frame
            aligned_telemetry[frame_idx, 0] = telemetry_features['blinker'][last_log_idx]
            aligned_telemetry[frame_idx, 1] = telemetry_features['brake'][last_log_idx]
            aligned_telemetry[frame_idx, 2] = telemetry_features['brake_computer'][last_log_idx]
            aligned_telemetry[frame_idx, 3] = telemetry_features['brake_user'][last_log_idx]
            aligned_telemetry[frame_idx, 4] = telemetry_features['cam1_ptr'][last_log_idx]
            aligned_telemetry[frame_idx, 5] = telemetry_features['car_accel'][last_log_idx]
            aligned_telemetry[frame_idx, 6] = telemetry_features['gas'][last_log_idx]
            aligned_telemetry[frame_idx, 7] = telemetry_features['speed'][last_log_idx]
            aligned_telemetry[frame_idx, 8] = telemetry_features['speed_abs'][last_log_idx]
            aligned_telemetry[frame_idx, 9] = telemetry_features['standstill'][last_log_idx]
            aligned_telemetry[frame_idx, 10] = telemetry_features['steering_angle'][last_log_idx]
            aligned_telemetry[frame_idx, 11] = telemetry_features['steering_torque'][last_log_idx]
            aligned_telemetry[frame_idx, 12] = telemetry_features['times'][last_log_idx]
        else:
            #if no log entry points to this frame, use values from previous frame
            #this should rarely happen if the data is properly structured
            if frame_idx > 0:
                aligned_telemetry[frame_idx] = aligned_telemetry[frame_idx - 1]
    
    return aligned_telemetry

def plot_frames(frame_sequence_np, n, max_display=5):
    #function to display a sequence of frames from a numpy array
    n_frames = min(n, len(frame_sequence_np))
    n_figures = math.ceil(n_frames / max_display)
    
    for fig_idx in range(n_figures):
        #calculate frame indices for current figure
        start_idx = fig_idx * max_display
        end_idx = min(start_idx + max_display, n_frames)
        frames_this_fig = end_idx - start_idx
        
        #create subplot grid
        fig, axes = plt.subplots(1, frames_this_fig, figsize=(15, 3))
        
        #handle case where only one frame is plotted
        if frames_this_fig == 1:
            axes = [axes]
        
        #plot each frame in current figure
        for i, ax in enumerate(axes):
            frame_idx = start_idx + i
            if frame_idx < n_frames:
                frame = frame_sequence_np[frame_idx]
                
                #convert from (C,H,W) to (H,W,C) format for display if needed
                if frame.shape[0] == 3 or frame.shape[0] == 1:
                    frame = np.transpose(frame, (1, 2, 0))
                
                #handle grayscale images
                if frame.shape[-1] == 1:
                    frame = frame.squeeze(-1)
                    ax.imshow(frame, cmap='gray')
                else:
                    ax.imshow(frame)
                
                ax.set_title(f"Frame {frame_idx}")
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()

def plot_telemetry(label_sequence_np, feature_index, feature_name):
    #extracts and plots a single telemetry feature from the labels array
    feature_data = label_sequence_np[:, feature_index]
    
    plt.figure(figsize=(12, 4))
    plt.plot(range(len(feature_data)), feature_data)
    
    #set labels and title
    plt.xlabel('Frame Number')
    plt.ylabel(feature_name)
    plt.title(f'{feature_name} Telemetry')
    
    #set y-axis limits based ONLY on this specific feature's min/max
    y_min, y_max = np.min(feature_data), np.max(feature_data)
    #add a small buffer for visual clarity
    y_range = y_max - y_min
    buffer = y_range * 0.05 if y_range > 0 else 0.1
    plt.ylim([y_min - buffer, y_max + buffer])
    
    #add grid for better readability
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_all_telemetry(label_sequence_np, sequence_name=None):
    #helper function to plot all telemetry features
    feature_names = [
        'Blinker', 'Brake (Combined)', 'Brake Computer', 'Brake User', 
        'Cam1 Ptr', 'Car Accel', 'Gas', 'Speed', 'Speed Abs', 
        'Standstill', 'Steering Angle', 'Steering Torque', 'Times'
    ]
    
    for i, name in enumerate(feature_names):
        print(f"Plotting {name}")
        plot_telemetry(label_sequence_np, i, name)
        
        #add a title with sequence name if provided
        if sequence_name:
            plt.suptitle(f"Sequence: {sequence_name}", fontsize=16)



#shape of output is a dictionary with 'train', 'val' and 'test' list
# processed_data = {
#     'train': [],
#     'val': [],
#     'test': []
# }
#each element in the list is the data for 1 video sample in the dictionary
# video_sample = {
#     'frames': frames, -> contains normalised input shape of frames/sequence_length x channels x height x width
#     'telemetry': aligned_telemetry, -> telemetry data that is aligned to frames with shape frames/sequence_length x 13
#     'file_name': file_name -> is the file name
# }
comma_ai_preprocessed = preprocess_comma_ai_data(comma_ai_data)
# plot_frames(comma_ai_preprocessed['train'][0]['frames'], 5)



# telemetry_feature_mapping = {
#     0: 'Blinker',           -> turn signal indicator
#     1: 'Brake (Combined)',  -> combined brake value 
#     2: 'Brake Computer',    -> commanded brake [0-4095]
#     3: 'Brake User',        -> user brake pedal depression [0-4095]
#     4: 'Cam1 Ptr',          -> camera frame index at this time
#     5: 'Car Accel',         -> car acceleration in m/s^2
#     6: 'Gas',               -> gas pedal position [0-1]
#     7: 'Speed',             -> vehicle speed in m/s
#     8: 'Speed Abs',         -> absolute vehicle speed in m/s
#     9: 'Standstill',        -> is the car stopped?
#     10: 'Steering Angle',   -> steering wheel angle
#     11: 'Steering Torque',  -> steering angle rate in deg/s
#     12: 'Times'             -> timestamps in seconds
# }
plot_telemetry(comma_ai_preprocessed['train'][0]['telemetry'], 10, 'Steering angle')


