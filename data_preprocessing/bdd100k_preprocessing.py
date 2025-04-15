from data_reading import bdd100k_data
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

#SPECIFY HERE AMOUNT OF FILES FROM EACH FOLDER YOU WANT TO INCLUDE IN TRAINING, VALIDATION AND TESTING
#========== limit of files ==========
MAX_LANE_TRAIN_FILES = 100 #limit number of lane segmentation training files out of 70,000
MAX_LANE_VAL_FILES = 50 #limit number of lane segmentation validation files out of 10,000
MAX_ENV_TRAIN_FILES = 100 #limit number of environment segmentation training files out of 2,976
MAX_TEST_FILES = 50 #limit number of test files out of 20,000

def normalise_path(path):
    #normalise path separators to be consistent
    return path.replace("\\", '/')

def preprocess_bdd100k_data(bdd100k_data):
    #create output dictionary structure with nested dictionaries for each data type and split
    processed_data = {
        'lane': {
            'train': {
                'raw': [],    #array of raw images for lane training
                'labels': []  #array of label images for lane training
            },
            'val': {
                'raw': [],    #array of raw images for lane validation
                'labels': []  #array of label images for lane validation
            }
        },
        'env': {
            'train': {
                'raw': [],    #array of raw images for environment training
                'labels': []  #array of label images for environment training
            }
        },
        'test': []            #array of test images (no labels)
    }
    
    #process lane segmentation data (train and val splits)
    process_lane_segmentation(bdd100k_data, processed_data)
    
    #process environment segmentation data (train split only)
    process_env_segmentation(bdd100k_data, processed_data)
    
    #process test images
    process_test_images(bdd100k_data, processed_data)
    
    return processed_data

def process_lane_segmentation(bdd100k_data, processed_data):
    #process lane segmentation training data
    print("Processing lane segmentation training data...")
    process_segmentation_split(
        bdd100k_data['raw']['train'], 
        bdd100k_data['lane']['train'],
        processed_data['lane']['train']['raw'],
        processed_data['lane']['train']['labels'],
        'lane', 'train',
        MAX_LANE_TRAIN_FILES
    )
    
    #process lane segmentation validation data
    print("Processing lane segmentation validation data...")
    process_segmentation_split(
        bdd100k_data['raw']['val'], 
        bdd100k_data['lane']['val'],
        processed_data['lane']['val']['raw'],
        processed_data['lane']['val']['labels'],
        'lane', 'val',
        MAX_LANE_VAL_FILES
    )

def process_env_segmentation(bdd100k_data, processed_data):
    #process environment segmentation training data (only train split available)
    print("Processing environment segmentation training data...")
    process_segmentation_split(
        bdd100k_data['raw']['train'], 
        bdd100k_data['env']['train'],
        processed_data['env']['train']['raw'],
        processed_data['env']['train']['labels'],
        'env', 'train',
        MAX_ENV_TRAIN_FILES
    )

def process_test_images(bdd100k_data, processed_data):
    #process test images (no segmentation masks)
    print("Processing BDD100K test images...")
    test_images = []
    
    #limit to MAX_TEST_FILES
    for image_path in bdd100k_data['raw']['test'][:MAX_TEST_FILES]:
        try:
            #normalise path
            image_path = normalise_path(image_path)
            
            #load image
            image = np.array(Image.open(image_path))
            
            #normalise to [0-1] range
            image = image.astype(np.float32) / 255.0
            
            #convert to channels-first format for compatibility with models
            image = np.transpose(image, (2, 0, 1))
            
            #append to collection
            test_images.append(image)
                
        except Exception as e:
            print(f"Error processing test image {image_path}: {e}")
    
    #convert list of images to numpy array with shape (n_samples, channels, height, width)
    if test_images:
        processed_data['test'] = np.array(test_images)
        print(f"Processed {len(test_images)} test images with shape {processed_data['test'].shape}")
    else:
        print("No test images were processed")

def get_base_filename(filename):
    #extract base part of filename before suffixes like _drivable_color.png
    #e.g., "00054602-3bf57337_drivable_color.png" -> "00054602-3bf57337"
    
    #remove file extension first
    base_name = os.path.splitext(filename)[0]
    
    #then split by underscore
    base_name = base_name.split('_')[0]
    
    return base_name

def process_segmentation_split(raw_paths, seg_paths, raw_output, labels_output, seg_type, split, max_files):
    #process segmentation data for a specific split
    #create mapping from base filenames to full paths for faster lookup
    raw_basename_to_path = {}
    for path in raw_paths:
        path = normalise_path(path)
        filename = os.path.basename(path)
        base_name = get_base_filename(filename)
        raw_basename_to_path[base_name] = path
    
    raw_images = []
    label_images = []
    matched_count = 0
    
    #limit to max_files
    for seg_path in seg_paths[:max_files]:
        try:
            #normalise path
            seg_path = normalise_path(seg_path)
            
            #get segmentation mask filename and base name
            seg_filename = os.path.basename(seg_path)
            base_name = get_base_filename(seg_filename)
            
            #find corresponding raw image
            raw_path = raw_basename_to_path.get(base_name)
            
            if raw_path:
                matched_count += 1
                
                #load raw image
                raw_image = np.array(Image.open(raw_path))
                
                #normalise to [0-1] range
                raw_image = raw_image.astype(np.float32) / 255.0
                
                #convert to channels-first format (C, H, W) for compatibility with models
                raw_image = np.transpose(raw_image, (2, 0, 1))
                
                #load segmentation mask
                label_image = np.array(Image.open(seg_path))
                
                #append to collections
                raw_images.append(raw_image)
                label_images.append(label_image)
                
        except Exception as e:
            print(f"Error processing {seg_type} {split} file {seg_path}: {e}")
    
    #convert lists to numpy arrays
    if raw_images and label_images:
        raw_output.extend(raw_images)
        labels_output.extend(label_images)
        print(f"Processed {matched_count} {seg_type} {split} image pairs")
        print(f"Raw image shape: {raw_images[0].shape}, Label shape: {label_images[0].shape}")
    else:
        print(f"No {seg_type} {split} images were processed")



def plot_bdd_samples(raw_images, label_images=None, n_samples=3):
    #visualise preprocessed bdd100k image samples
    
    #determine actual number of samples to plot
    n = min(n_samples, 5, len(raw_images))
    
    #iterate through samples to plot
    for i in range(n):
        if label_images is not None:
            #create figure with two subplots for raw and label images
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            #plot raw image
            raw_img = raw_images[i]
            #convert from channels-first to channels-last format
            if raw_img.shape[0] == 3 and len(raw_img.shape) == 3:
                raw_img = np.transpose(raw_img, (1, 2, 0))
            axes[0].imshow(raw_img)
            axes[0].set_title(f"Raw Image [{i}]")
            axes[0].axis('off')
            
            #plot label image
            label_img = label_images[i]
            #handle grayscale vs color labels
            if len(label_img.shape) == 2:
                axes[1].imshow(label_img, cmap='gray')
            else:
                axes[1].imshow(label_img)
            axes[1].set_title(f"Label Image [{i}]")
            axes[1].axis('off')
            
        else:
            #create figure with single subplot for test image
            fig, ax = plt.subplots(figsize=(8, 8))
            
            #plot test image
            raw_img = raw_images[i]
            #convert from channels-first to channels-last format
            if raw_img.shape[0] == 3 and len(raw_img.shape) == 3:
                raw_img = np.transpose(raw_img, (1, 2, 0))
            ax.imshow(raw_img)
            ax.set_title(f"Test Image [{i}]")
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

#shape of output is:
# processed_data = {
#     'lane': {
#         'train': {
#             'raw': [array(...), array(...), ...],    #list of raw images with shape (3, H, W) - channels first
#             'labels': [array(...), array(...), ...]  #list of label masks with shape (H, W) or (H, W, C)
#         },
#         'val': {
#             'raw': [...],
#             'labels': [...]
#         }
#     },
#     'env': {
#         'train': {
#             'raw': [...],
#             'labels': [...]
#         }
#     },
#     'test': ndarray(...)  #numpy array with shape (n_samples, 3, H, W) - batch of images
# }
#access examples:
# - lane_raw_samples = processed_data['lane']['train']['raw'][0:10]  #first 10 lane training images
# - lane_labels_samples = processed_data['lane']['train']['labels'][0:10]  #corresponding labels
# - env_raw_samples = processed_data['env']['train']['raw'][5]  #6th environment training image
# - env_labels_samples = processed_data['env']['train']['labels'][5]  #corresponding label
# - test_samples = processed_data['test'][0:5]  #first 5 test images

#example usage:
bdd100k_preprocessed = preprocess_bdd100k_data(bdd100k_data)
print("Data preprocessing complete.")
print(f"Lane training samples: {len(bdd100k_preprocessed['lane']['train']['raw'])}")
print(f"Lane validation samples: {len(bdd100k_preprocessed['lane']['val']['raw'])}")
print(f"Environment training samples: {len(bdd100k_preprocessed['env']['train']['raw'])}")
print(f"Test samples: {len(bdd100k_preprocessed['test'])}")


plot_bdd_samples(bdd100k_preprocessed['lane']['train']['raw'], bdd100k_preprocessed['lane']['train']['labels'])
plot_bdd_samples(bdd100k_preprocessed['env']['train']['raw'], bdd100k_preprocessed['env']['train']['labels'])