import os
import numpy as np
import h5py
from gui_alt.steering_trajectory import draw_trajectory_annotation
import albumentations as A

#allowed telemetry labels
TELEMETRY_LABELS = ['steering_angle','times','speed', 'car_accel', 'speed_abs']
FPS_DEFAULT = 20
#default speed value when not available in telemetry
DEFAULT_SPEED = 25.0  #in km/h

#returns true if file name ends with .h5
def is_h5(path):
    return path.lower().endswith('.h5')

#swap camera path with labels to directly get corresponding telemetry
def get_label_path(camera_path):
    return camera_path.replace('camera','labels')

#helper function to return video file as numpy array in shape F x C x H x W
def load_video(file_path):
    with h5py.File(file_path,'r') as f:
        arr = np.asarray(f['X'])
    return arr

#this helper aligns telemetry data to video frames
def align_telemetry(data, ptr):
    ptr_int = ptr.astype(int)
    max_f = ptr_int.max()
    out = np.empty(max_f+1,dtype=data.dtype)
    for fidx in range(max_f+1):
        hits = np.where(ptr_int==fidx)[0]
        if hits.size>0:
            out[fidx] = data[hits[-1]]
        else:
            out[fidx] = out[fidx-1] if fidx>0 else 0
    return out

#this helper function loads telemetry and aligns it to number of frames, returns dictionary of labels with each value's shape being 1 dimensional numpy array
def load_telemetry(file_path, labels, fps=FPS_DEFAULT):
    tel = {}
    with h5py.File(file_path,'r') as f: #open file
        ptr = np.asarray(f['cam1_ptr'])

        for lab in labels:
            if lab=='times': #seperate times label
                continue

            if lab not in f: #skip invalid labels
                continue

            raw = np.asarray(f[lab])
            if lab=='steering_angle':
                raw = raw/10 #remember to scale steering angles

            #no scaling for car_accel
            tel[lab] = align_telemetry(raw,ptr)

    frame_max = max(v.shape[0] for v in tel.values()) if tel else 0
    tel['times'] = np.arange(frame_max)/fps #handle time here
    
    #extract speed_abs as speed feature
    if 'speed_abs' in f:
        raw_speed = np.asarray(f['speed_abs'])
        tel['speed_abs'] = align_telemetry(raw_speed, ptr)
    
    return tel

#load camera frames and telemetry
def load_camera_and_labels(camera_path):
    #validate camera path
    if not is_h5(camera_path):
        raise ValueError('invalid file type')
    
    if not os.path.isfile(camera_path):
        raise FileNotFoundError('camera not found')
    
    label_path = get_label_path(camera_path)

    if not os.path.isfile(label_path):
        raise FileNotFoundError('labels not found')
    
    #load video frames F x C x H x W
    frames = load_video(camera_path)
    telemetry = load_telemetry(label_path, TELEMETRY_LABELS)
    return frames,telemetry

#get total frames
def get_frame_count(frames):
    return frames.shape[0] #frames shape F x C x H x W

#get frame at idx as h x w x c uint8
def get_frame(frames, idx):
    idx = max(0,min(idx,get_frame_count(frames)-1)) #ensure index is within number of frames
    frame = frames[idx].transpose(1,2,0) #transpose to H x W x C from C x H x W
    return np.clip(frame,0,255).astype(np.uint8) #cast as uint8 for memory efficiency

#get speed at specific frame if available
def get_speed(telemetry, idx):
    #prioritise speed_abs over speed if available
    if telemetry and 'speed_abs' in telemetry:
        return telemetry['speed_abs'][idx]
    
    elif telemetry and 'speed' in telemetry:
        return telemetry['speed'][idx]
    
    return DEFAULT_SPEED

#get the current acceleration
def get_car_acceleration(telemetry, idx):
    #get car acceleration if available
    if telemetry and 'car_accel' in telemetry:
        return telemetry['car_accel'][idx]
    return 0.0  #default acceleration if not available

#apply trajectory overlay to frame
def apply_trajectory_overlay(frame, telemetry, idx, true_angle, sim_angle):
    #get speed for trajectory calculation
    speed = get_speed(telemetry, idx)
    
    #apply trajectory overlay
    return draw_trajectory_annotation(frame, speed, true_angle, sim_angle)

#apply brightness augmentation to make frame brighter
def apply_light_augmentation(frame, intensity=0.0):
    if intensity <= 0.0:
        return frame
    
    #create augmentation with brightness increase
    aug = A.RandomBrightnessContrast(
        brightness_limit=(intensity, intensity),
        contrast_limit=0,
        p=1.0
    )
    
    #apply augmentation
    augmented = aug(image=frame)['image']
    return augmented

#apply brightness augmentation to make frame darker
def apply_dim_augmentation(frame, intensity=0.0):
    if intensity <= 0.0:
        return frame
    
    #create augmentation with brightness decrease
    aug = A.RandomBrightnessContrast(
        brightness_limit=(-intensity, -intensity),
        contrast_limit=0,
        p=1.0
    )
    
    #apply augmentation
    augmented = aug(image=frame)['image']
    return augmented

#apply realistic camera grain noise to frame, note this is not the same noise augmentation used during training, but regardless showcases model resilience as it was never trained on this type of noisy input
def apply_noise_augmentation(frame, intensity=0.0):
    if intensity <= 0.0:
        return frame
    
    #create a copy of the frame to avoid modifying the original
    result = frame.copy().astype(np.float32)
    
    #scale intensity for better control (0.1 is quite visible)
    scaled_intensity = intensity * 80.0
    
    #generate colored noise that preserves image characteristics
    #colour-aware noise - brighter areas get more noise
    noise = np.random.normal(0, scaled_intensity, frame.shape).astype(np.float32)
    
    #apply the noise in a way that preserves colours
    #make noise proportional to pixel values (darker areas get less noise)
    colour_factor = (result / 255.0) * 0.9 + 0.5  #prevents completely black areas from having no noise
    
    #apply noise with colour preservation
    result = result + noise * colour_factor * 0.8 * np.sqrt(result/255.0)
    
    #ensure values stay in valid range
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

#apply all augmentations to a frame based on slider values
def apply_augmentations(frame, light_value=0.0, dim_value=0.0, noise_value=0.0):
    #apply augmentations in sequence
    augmented_frame = frame.copy()
    
    if light_value > 0.0:
        augmented_frame = apply_light_augmentation(augmented_frame, light_value)
    
    if dim_value > 0.0:
        augmented_frame = apply_dim_augmentation(augmented_frame, dim_value)
    
    if noise_value > 0.0:
        augmented_frame = apply_noise_augmentation(augmented_frame, noise_value)
    
    return augmented_frame

#process frame for display with all possible modifications (augmentations and trajectory)
def process_frame(frames, idx, telemetry=None, true_angle=0.0, sim_angle=0.0, 
                  show_trajectory=False, light_value=0.0, dim_value=0.0, noise_value=0.0):
    #get the base frame
    frame = get_frame(frames, idx)
    
    #apply augmentations first
    frame = apply_augmentations(frame, light_value, dim_value, noise_value)
    
    #apply trajectory overlay if enabled
    if show_trajectory and telemetry is not None:
        frame = apply_trajectory_overlay(frame, telemetry, idx, true_angle, sim_angle)
    
    return frame