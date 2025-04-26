import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

#THIS CODE WAS USE TO GENERATE POLYGON LABELS FOR LANE SEGMENTATION

# #find contours from binary mask
# def find_contours_from_mask(mask):
#     contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     #filter small contours (noise)
#     min_contour_area = 100
#     contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
#     #simplify contours to reduce number of points
#     simplified_contours = []
#     for contour in contours:
#         epsilon = 0.005 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#         simplified_contours.append(approx)
    
#     return simplified_contours

# #normalise coordinates to 0-1 range
# def normalise_coordinates(contours, img_width, img_height):
#     normalised_contours = []
    
#     for contour in contours:
#         #reshape contour to remove unnecessary dimension
#         contour = contour.reshape(-1, 2)
        
#         #normalise coordinates
#         normalised = contour.astype(float)
#         normalised[:, 0] /= img_width
#         normalised[:, 1] /= img_height
        
#         normalised_contours.append(normalised)
    
#     return normalised_contours

# #processing function
# def convert_masks(input_dir, output_dir):
#     #ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)
    
#     #get all png files in the directory
#     mask_files = glob(os.path.join(input_dir, "*.png"))
#     count = 0
    
#     #add tqdm progress bar
#     for mask_file in tqdm(mask_files, desc=f"Processing {os.path.basename(input_dir)}"):
#         #get filename without extension
#         filename = os.path.basename(mask_file)
#         filename_without_ext = os.path.splitext(filename)[0]
        
#         #read the image
#         mask = cv2.imread(mask_file)
#         img_height, img_width = mask.shape[:2]
        
#         #find blue pixels (adjacent lanes) - BGR format [255, 0, 0]
#         blue_mask = np.all(mask == [255, 0, 0], axis=2)
        
#         #find red pixels (current lane) - BGR format [0, 0, 255]
#         red_mask = np.all(mask == [0, 0, 255], axis=2)
        
#         #find contours for each class
#         blue_contours = find_contours_from_mask(blue_mask)  #class 0: adjacent lanes
#         red_contours = find_contours_from_mask(red_mask)    #class 1: current lane
        
#         #normalise coordinates
#         norm_blue_contours = normalise_coordinates(blue_contours, img_width, img_height)
#         norm_red_contours = normalise_coordinates(red_contours, img_width, img_height)
        
#         #create the output path for the text file
#         output_path = os.path.join(output_dir, f"{filename_without_ext}.txt")
        
#         #write to YOLO format file
#         with open(output_path, 'w') as f:
#             #write blue contours (class 0 - adjacent lanes)
#             for contour in norm_blue_contours:
#                 line = "0 " + " ".join([f"{x} {y}" for x, y in contour])
#                 f.write(line + "\n")
            
#             #write red contours (class 1 - current lane)
#             for contour in norm_red_contours:
#                 line = "1 " + " ".join([f"{x} {y}" for x, y in contour])
#                 f.write(line + "\n")
        
#         count += 1
    
#     return count

# #paths
# datasplits = ["test", "train", "val"]
# base_path = "datasets/BDD100K_Lane"

# #process each datasplit
# total_files = 0
# for split in datasplits:
#     total_split = 0
#     input_dir = f"{base_path}/{split}/masks_original"
#     output_dir = f"{base_path}/{split}/labels"
    
#     processed = convert_masks(input_dir, output_dir)
#     print(f"processed {processed} files for {split}")
#     total_split += processed
#     print(f"total for {split}: {total_split}")
#     total_files += processed

# print(f"total files processed: {total_files}")


#THIS CODE WAS USE TO GENERATE POLYGON LABELS FOR ENV SEGMENTATION

#define the 20 unique BGR colors and their class IDs
COLOR_CLASSES = [
    ((0, 0, 0), 0),           #black
    ((0, 0, 255), 1),         #red
    ((0, 220, 220), 2),       #yellow-orange
    ((30, 170, 250), 3),      #light orange
    ((32, 11, 119), 4),       #reddish
    ((35, 142, 107), 5),      #greenish
    ((60, 20, 220), 6),       #pinkish
    ((70, 0, 0), 7),          #dark blue
    ((70, 70, 70), 8),        #dark gray
    ((100, 60, 0), 9),        #dark cyan
    ((100, 80, 0), 10),       #dark cyan/green
    ((128, 64, 128), 11),     #purple
    ((142, 0, 0), 12),        #dark blue
    ((152, 251, 152), 13),    #light green
    ((153, 153, 153), 14),    #gray
    ((153, 153, 190), 15),    #light purple
    ((156, 102, 102), 16),    #blue-gray
    ((180, 130, 70), 17),     #light blue
    ((230, 0, 0), 18),        #blue
    ((232, 35, 244), 19)      #pink/magenta
]

#find contours from binary mask
def find_contours_from_mask(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #filter small contours (noise)
    min_contour_area = 100
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    #simplify contours to reduce number of points
    simplified_contours = []
    for contour in contours:
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        simplified_contours.append(approx)
    
    return simplified_contours

#normalise coordinates to 0-1 range
def normalise_coordinates(contours, img_width, img_height):
    normalised_contours = []
    
    for contour in contours:
        #reshape contour to remove unnecessary dimension
        contour = contour.reshape(-1, 2)
        
        #normalise coordinates
        normalised = contour.astype(float)
        normalised[:, 0] /= img_width
        normalised[:, 1] /= img_height
        
        normalised_contours.append(normalised)
    
    return normalised_contours

#processing function
def convert_masks(input_dir, output_dir):
    #ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    #get all png files in the directory
    mask_files = glob(os.path.join(input_dir, "*.png"))
    count = 0
    
    #add tqdm progress bar
    for mask_file in tqdm(mask_files, desc=f"Processing {os.path.basename(input_dir)}"):
        #get filename without extension
        filename = os.path.basename(mask_file)
        filename_without_ext = os.path.splitext(filename)[0]
        
        #read the image
        mask = cv2.imread(mask_file)
        img_height, img_width = mask.shape[:2]
        
        #create the output path for the text file
        output_path = os.path.join(output_dir, f"{filename_without_ext}.txt")
        
        #open file for writing
        with open(output_path, 'w') as f:
            #process each color class
            for color, class_id in COLOR_CLASSES:
                #find pixels matching this color
                class_mask = np.all(mask == color, axis=2)
                
                #find contours for this class
                contours = find_contours_from_mask(class_mask)
                
                #normalize coordinates
                norm_contours = normalise_coordinates(contours, img_width, img_height)
                
                #write contours to file
                for contour in norm_contours:
                    line = f"{class_id} " + " ".join([f"{x} {y}" for x, y in contour])
                    f.write(line + "\n")
        
        count += 1
    
    return count

#visualisation function
def visualise_polygons(mask_path, output_path):
    #read the original mask
    mask = cv2.imread(mask_path)
    img_height, img_width = mask.shape[:2]
    
    #create a blank image for visualisation
    vis_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    #random colors for visualisation (different from the class colors)
    vis_colors = np.random.randint(0, 255, (20, 3), dtype=np.uint8)
    
    #process each color class
    for i, (color, class_id) in enumerate(COLOR_CLASSES):
        #find pixels matching this color
        class_mask = np.all(mask == color, axis=2)
        
        #find contours
        contours = find_contours_from_mask(class_mask)
        
        #draw contours on visualisation image
        cv2.drawContours(vis_image, contours, -1, vis_colors[i].tolist(), thickness=1)
    
    #save visualisation image
    cv2.imwrite(output_path, vis_image)
    
    #display using matplotlib
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    plt.title('Original Mask')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.title('Contours')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_comparison.png'))
    plt.close()
    
    return vis_image

#paths
datasplits = ["test", "train", "val"]
base_path = "datasets/BDD100K_Env"

#process each datasplit
total_files = 0
for split in datasplits:
    total_split = 0
    input_dir = f"{base_path}/{split}/masks_original"
    output_dir = f"{base_path}/{split}/labels"
    
    processed = convert_masks(input_dir, output_dir)
    print(f"processed {processed} files for {split}")
    total_split += processed
    print(f"total for {split}: {total_split}")
    total_files += processed

print(f"total files processed: {total_files}")

#create a visualisation for one sample image
if total_files > 0:
    sample_split = datasplits[0]
    sample_masks = glob(os.path.join(base_path, sample_split, "masks_original", "*.png"))
    if sample_masks:
        sample_mask = sample_masks[0]
        vis_output = os.path.join(base_path, f"sample_visualisation.png")
        visualise_polygons(sample_mask, vis_output)
        print(f"created visualisation: {vis_output}")