import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import random

#used to find the number of unique pixel segmentation classes

def find_unique_colors(directory, num_images=10):
    #get all png files in the directory
    mask_files = glob(os.path.join(directory, "*.png"))
    
    #check if we have enough files
    if len(mask_files) < num_images:
        num_images = len(mask_files)
        print(f"warning: only {num_images} files found")
    
    #randomly sample if we have more images than requested
    if len(mask_files) > num_images:
        mask_files = random.sample(mask_files, num_images)
    
    #set to store unique colors
    unique_colors = set()
    count = 0
    #process each mask file
    for mask_file in tqdm(mask_files, desc="Processing masks"):
        #read the image
        mask = cv2.imread(mask_file)
        
        if count == 0:
            print(f"shape: {mask.shape}")

        #reshape to get all pixels as rows
        pixels = mask.reshape(-1, 3)

        if count == 0:
            print(f"Pixel shape: {pixels.shape}")
        
        #convert to tuples for hashability
        pixel_tuples = [tuple(pixel) for pixel in pixels]
        
        #add unique colors to our set
        unique_colors.update(pixel_tuples)

        count += 1
    
    #print results
    print(f"\nfound {len(unique_colors)} unique colors across {num_images} images")
    
    #print each unique color
    for color in sorted(unique_colors):
        print(f"BGR: {color}")
    
    return unique_colors

if __name__ == "__main__":
    #directory containing mask images
    # mask_dir = r"datasets\BDD100K_Lane\train\masks_original"
    mask_dir = r"datasets\BDD100K_Env\train\masks_original"
    
    #number of images to process
    num_samples = 10
    
    #find unique colors
    find_unique_colors(mask_dir, num_samples)