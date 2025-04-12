import h5py #import hdf5 library
import os #import os library for operating system dependent functionality eg path joining
import copy #import copy library for copying objects
import sys #import sys library for system-specific parameters and functions

#========== dataset paths ==========
comma_ai_path = "././dataset/Comma_ai_dataset" #path to comma ai dataset
bdd100k_path = "././dataset/BDD100k_dataset" #path to bdd100k dataset

#========== Check paths are correct ==========
#use to check if a specified path is correct and contains the required content
def check_path_content(directory):
    #check specified path exists and contains files
    try:
        directory_content = os.listdir(directory) #returns names of content (folders files etc) in directory as a list
        print("="*10,f"Content in {directory}", "="*10)

        for item in directory_content:
            print(f"{item}")

        if not directory_content:
            print("(Directory is empty)")

    #handle exceptions
    except FileNotFoundError:
        print(f"Directory not found: {directory}") 

    except PermissionError:
        print(f"Permission denied for directory: {directory}") 
 
    except Exception as e:
        print(f"An error occurred checking {directory}: {e}") 


#uncomment to check dataset folder paths are correct:
#print("--- Verifying Dataset Paths ---")
#if not check_path_content(comma_ai_path):
#    sys.exit("Exiting due to Comma.ai path issue")
#if not check_path_content(bdd100k_path):
#    sys.exit("Exiting due to BDD100k path issue")
#print("--- Dataset Paths Verified ---\n")


#========== variables for storing dataset filenames & paths ==========
datasplit = {
    'train': [], #training filenames/paths list
    'val': [], #validation filenames/paths list
    'test': [] #testing filenames/paths list
}

#use deepcopy to avoid modifying the original datasplit variable, it is just a template
comma_ai_data = {
    'camera': copy.deepcopy(datasplit),
    'labels': copy.deepcopy(datasplit)
}

bdd100k_data = {
    'raw': copy.deepcopy(datasplit), #raw images corresponding to labels
    'lane': copy.deepcopy(datasplit), #lane segmentation labels
    'env': copy.deepcopy(datasplit) #environment segmentation labels
}

#remove splits that dont exist for specific label types
bdd100k_data['lane'].pop('test', None) #lane seg has no specific test folder, will use test folder from raw
bdd100k_data['env'].pop('test', None) #env seg has no specific test folder, will use test folder from raw
bdd100k_data['env'].pop('val', None) #env seg has no specific val folder

#========== read Comma ai dataset filenames/paths ==========
print("--- Reading Comma.ai Data ---") 
for split in ['train', 'val', 'test']:
    cam_path = os.path.join(comma_ai_path, split, 'camera') #path for camera files
    label_path = os.path.join(comma_ai_path, split, 'labels') #path for label files

    #store full paths for camera files--------
    if os.path.exists(cam_path) and os.path.isdir(cam_path): #check path exists and is directory
        try:
            #list comprehension to get full paths of files only 
            full_paths = [os.path.join(cam_path, f) for f in os.listdir(cam_path) if os.path.isfile(os.path.join(cam_path, f))]
            comma_ai_data['camera'][split] = full_paths #assign list of paths
            #print(f"Comma.ai Camera [{split}]: Stored {len(full_paths)} full paths") 
        except Exception as e:
            print(f"Error reading Comma.ai camera files for '{split}': {e}") 
    else:
         print(f"Warning: Comma.ai camera path not found or not a directory: '{cam_path}'") 

    #store full paths for label files--------
    if os.path.exists(label_path) and os.path.isdir(label_path):
        try:
            full_paths = [os.path.join(label_path, f) for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))]
            comma_ai_data['labels'][split] = full_paths 
            #print(f"Comma.ai Labels [{split}]: Stored {len(full_paths)} full paths") 
        except Exception as e:
            print(f"Error reading Comma.ai label files for '{split}': {e}") 
    else:
        print(f"Warning: Comma.ai label path not found or not a directory: '{label_path}'") 
print("--- Finished Reading Comma.ai Data ---\n") 


#========== read BDD100K dataset filenames/paths ==========
#raw images split
raw_env_folder = 'environment_seg'
raw_lane_folder = 'lane_seg'

print("--- Reading BDD100k Data ---") 
#raw images
for split in ['train', 'val', 'test']:
    paths_to_read = [] #list to hold paths for the current split
    if split == 'test':
        #test images are directly under raw_images/test
        raw_test_path = os.path.join(bdd100k_path, 'raw_images', 'test')
        paths_to_read.append(raw_test_path)
    else:
        #train/val images are under raw_images/environment_seg/split and raw_images/lane_seg/split
        paths_to_read.append(os.path.join(bdd100k_path, 'raw_images', raw_env_folder, split))
        paths_to_read.append(os.path.join(bdd100k_path, 'raw_images', raw_lane_folder, split))

    split_raw_paths = [] #collect paths for this split first
    for raw_path in paths_to_read:
        if os.path.exists(raw_path) and os.path.isdir(raw_path): #check path exists and is directory
            try:
                #list comprehension to get full paths of files only
                full_paths = [os.path.join(raw_path, f) for f in os.listdir(raw_path) if os.path.isfile(os.path.join(raw_path, f))]
                split_raw_paths.extend(full_paths) #add paths from this source folder
                #print(f"BDD100k Raw [{split}]: Added {len(full_paths)} paths from '{raw_path}'")
            except Exception as e:
                 print(f"Error reading BDD100k raw files from '{raw_path}': {e}") 
        else:
            #only warn if the path was expected
             if not (split == 'test' and (raw_env_folder in raw_path or raw_lane_folder in raw_path)):
                 print(f"Warning: BDD100k raw path not found or not a directory: '{raw_path}'") 
    bdd100k_data['raw'][split] = split_raw_paths #assign collected paths for the split


#lane labels
print("\nReading BDD100k Lane Labels...") 
for split in ['train', 'val']:
    lane_path = os.path.join(bdd100k_path, 'lane_labels', split) 
    if os.path.exists(lane_path) and os.path.isdir(lane_path): 
        try:
            full_paths = [os.path.join(lane_path, f) for f in os.listdir(lane_path) if os.path.isfile(os.path.join(lane_path, f))]
            bdd100k_data['lane'][split] = full_paths 
        except Exception as e:
             print(f"Error reading BDD100k lane label files for '{split}': {e}") 
    else:
        print(f"Warning: BDD100k lane label path not found or not a directory: '{lane_path}'") 


#env labels (only train)
print("\nReading BDD100k Environment Labels...")
env_path = os.path.join(bdd100k_path, 'environment_seg_labels', 'train') 
if os.path.exists(env_path) and os.path.isdir(env_path): 
     try:
        full_paths = [os.path.join(env_path, f) for f in os.listdir(env_path) if os.path.isfile(os.path.join(env_path, f))]
        bdd100k_data['env']['train'] = full_paths 
     except Exception as e:
        print(f"Error reading BDD100k env label files for 'train': {e}") 
else:
     print(f"Warning: BDD100k env label path not found or not a directory: '{env_path}'") 
print("--- Finished Reading BDD100k Data ---\n") 


#========== check file counts ==========
def get_file_counts(dataset_dict, name):
    print(f"\n--- File Counts for {name} ---") 

    for key, datasplits in dataset_dict.items(): 
        print("="*15, f"{key} Count", "="*15) 

        for split, files in datasplits.items(): 
            print(f"- {split}: {len(files)}") 

    #print a dynamic separator line based on the key length
    if dataset_dict: 
        last_key = list(dataset_dict.keys())[-1] #get last key for formatting
        print("-" * (32 + len(last_key))) #print separator

get_file_counts(bdd100k_data, "BDD100k")
print("\n") 
get_file_counts(comma_ai_data, "Comma.ai")


print(f"First Comma.ai camera train path: {comma_ai_data['camera']['train'][0]}")
print(f"First BDD100k raw train path: {bdd100k_data['raw']['train'][0]}")
print(f"First BDD100k lane train path: {bdd100k_data['lane']['train'][0]}")
print(f"First BDD100k env train path: {bdd100k_data['env']['train'][0]}")


#TO DO
# - organise into numpy data split