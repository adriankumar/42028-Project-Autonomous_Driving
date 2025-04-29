import numpy as np
import h5py 
import os 

#dataset paths ~ manually need to add 'camera' or 'label' afterwards
train_path = r"datasets\Comma_ai_dataset\train"
val_path = r"datasets\Comma_ai_dataset\val"
test_path = r"datasets\Comma_ai_dataset\test"

#file names in index of smallest to largest, they are the same for camera and label 

train_files = ['2016-06-08--11-46-01.h5', #2.63GB <-- smallest
               '2016-02-08--14-56-28.h5', #3.80GB
               '2016-06-02--21-39-29.h5', #6.44GB   ..to
               '2016-02-02--10-16-58.h5', #8.05GB
               '2016-03-29--10-50-20.h5', #11.2GB
               '2016-02-11--21-32-47.h5'] #12.3GB <-- largest
               
val_files = ['2016-04-21--14-48-08.h5' #4.39GB <-- smallest
             '2016-01-30--11-24-51.h5'] #7.61GB <-- largest

test_files = ['2016-01-31--19-19-25.h5', #2.92GB <-- smallest
              '2016-05-12--22-20-00.h5', #7.47GB        ..to
              '2016-01-30--13-46-00.h5'] #8.49GB <-- largest

telemetry_keys = ['steering_angle', 'steering_torque', 'speed', 'speed_abs', 'car_accel', 'brake', 'brake_user', 'brake_computer', 'standstill', 'times']


def get_frame_length(datasplit_path, file_name):
    camera_file = f"{datasplit_path}\camera\{file_name}"

    with h5py.File(camera_file, 'r') as video:
        shape = np.array(video['X']).shape #is shape frames x colours x height x width

    print(f"{file_name} has {shape[0]} frames")

    return shape[0]

#aligns telemetry data to num of frames by selecting the last index from telemetry to include as value at time step t in frame
def align_telemetry(telemetry_dict, num_frames, cam1_ptr):
    aligned_telemetry = {}

    for key, _ in telemetry_dict.items():
        aligned_data = np.zeros(num_frames, dtype=np.float32) #one dimensional of shape frames

        for frame_index in range(num_frames):
            # telemetry_indicies = np.where(cam1_ptr == frame_index)[0] #returns tuple, index [0] gives us the index list
            telemetry_indicies = np.atleast_1d(cam1_ptr == frame_index).nonzero()[0]

            #safety handling if telemetry indicies doesnt exist
            if len(telemetry_indicies) > 0: #if it exists, use last index to retrieve aligned value
                last_index = telemetry_indicies[-1]
                aligned_data[frame_index] = telemetry_dict[key][last_index]

            elif frame_index > 0: #otherwise use previous value
                aligned_data[frame_index] = aligned_data[frame_index - 1]
        
        aligned_telemetry[key] = aligned_data
    
    return aligned_telemetry


def normalise_telemetry(telemetry_aligned, names):
    for name in names:
        if name == "steering_angle":
            telemetry_aligned[name] = telemetry_aligned[name] / 10.0 #steering angles are scaled by 10 in dataset, normalise by dividing by 10
    
    #add other known normalisation here 

    return telemetry_aligned
                

def process_h5_file(datasplit_path, file_name, telemetry_names, start=None, end=None): #returns video_data & aligned_telemtry
    camera_file = f"{datasplit_path}\camera\{file_name}" 
    telemetry_file = f"{datasplit_path}\labels\{file_name}"

    print(f"Camera file is valid: {os.path.isfile(camera_file)}")
    print(f"Label file is valid: {os.path.isfile(telemetry_file)}")

    telemetry_data = {} #stores names and the 1d array as values

    if not os.path.isfile(camera_file) or not os.path.isfile(telemetry_file):
        raise ValueError(f"Either {camera_file} or {telemetry_file} is invalid")
    
    #check if name is valid
    for name in telemetry_names:
        if name not in telemetry_keys:
            raise ValueError(f"{name} does not exist or is not used in {telemetry_keys}")

    #read camera file
    with h5py.File(camera_file, 'r') as video:
        video_data = np.array(video['X']) #is shape frames x colours x height x width
        frames = video_data.shape[0] 

        #handling start and end indicies
        if start is None and end is None:
            start = 0
            end = frames-1

        elif (start + end - 1) >= frames:
            raise ValueError(f"Either index {start} for start or index {end} for end is not valid; there are {frames} frames select from indices: [0,..{frames-1}]")

        video_normalised = video_data.astype(np.float32) / 225.0 #normalise pixel values
    
    #read telemetry data
    with h5py.File(telemetry_file, 'r') as telemetry:
        for name in telemetry_names:
            data = np.array(telemetry[name]) #get telemetry data whos shape is a unique 1d sequence length due to sampling frequency mismatch
            telemetry_data[name] = data

        cam_ptr = telemetry['cam1_ptr'] #get ptr for alinging telemetry data
    
    #align telemetry data before splicing
    telemetry_data = align_telemetry(telemetry_data, frames, cam_ptr) #should be same shape as frames

    telemetry_data = normalise_telemetry(telemetry_data, telemetry_names) #scale the certain telemetry values by a fixed amount for true values i.e steering angle is scaled by 10 so we divide the original angles by 10

    video_spliced = video_normalised[start:end]
    telemetry_spliced = {}

    for key in telemetry_data.keys():
        telemetry_spliced[key] = telemetry_data[key][start:end]

    return video_spliced, telemetry_spliced #return video_data (f x c x h x w but normalised) and aligned telemetry dictionary each with shape n_frames

#example use case
# video_data, telemetry_data = process_h5_file(datasplit_path=train_path, file_name=train_files[0], telemetry_names=['steering_angle'], start=2500, end=6000)
# print(telemetry_data['steering_angle'].shape)
# print(video_data.shape)
