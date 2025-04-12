import h5py
import numpy as np
import matplotlib.pyplot as plt

#USE THIS CODE TO TEST:
# - modules in conda virutal environment are recognised (h5py, numpy and matplotlib)
# - successful loading of a sample data
# - dataset sample contains correct data to access information
# Once this code successfully runs, you can use the 'video_display.py' file

# Open the files
camera_file = h5py.File(r'dataset\Comma_ai_dataset\train\camera\2016-06-08--11-46-01.h5', 'r')
log_file = h5py.File(r'dataset\Comma_ai_dataset\train\labels\2016-06-08--11-46-01.h5', 'r')

# See what's inside
print("Camera file contents:")
print(list(camera_file.keys()))
print("\nLog file contents:")
print(list(log_file.keys()))

# Access camera frames
# Each frame in X is stored as (3, 160, 320) - (channels, height, width)
frames = camera_file['X']  
print(f"\nCamera data shape: {frames.shape}")

# Access a single frame (convert from HDF5 dataset to numpy array)
frame = frames[18000][:].transpose(1, 2, 0)  # Rearrange to (height, width, channels)
print(f"Single frame shape: {frame.shape}")

#Display a frame
plt.imshow(frame)
plt.axis('off')
plt.show()

#Access steering angles 
steering_angles = log_file['steering_angle'][:]
speed = log_file['speed'][:]
print(f"\nSteering angles shape: {steering_angles.shape}, example value: {steering_angles[0]}")
print(f"\nSpeed: shape: {speed.shape}, example value: {speed[0]}")

# Close the files
camera_file.close()
log_file.close()