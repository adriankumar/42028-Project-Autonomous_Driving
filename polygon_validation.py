import os
from glob import glob
from tqdm import tqdm

#This script was to check the polygon labels generated in the dataset for lane segmentatation

#function to check single label file
def check_polygon_file(file_path):
    invalid_polygons = []
    
    if not os.path.exists(file_path):
        return [f"File not found: {file_path}"]
        
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 7:  #class_id + at least 3 points (6 coordinates)
                invalid_polygons.append(f"Polygon {i+1}: Too few parts ({len(parts)})")
                continue
                
            try:
                class_id = int(parts[0])
                points = [float(p) for p in parts[1:]]
                
                #check if all coordinates are valid (between 0-1)
                valid_coords = all(0 <= p <= 1 for p in points)
                #check if there are enough points (at least 3 points = 6 coordinates)
                enough_points = len(points) >= 6 and len(points) % 2 == 0
                
                if not valid_coords:
                    invalid_polygons.append(f"Polygon {i+1}: Contains coordinates outside 0-1 range")
                if not enough_points:
                    invalid_polygons.append(f"Polygon {i+1}: Not enough points or odd number of coordinates")
            except Exception as e:
                invalid_polygons.append(f"Polygon {i+1}: Error parsing: {str(e)}")
    except Exception as e:
        return [f"Error reading file: {str(e)}"]
        
    return invalid_polygons

#function to check all label files in a directory
def check_all_labels(base_path):
    dirs = ["test", "train", "val"]
    invalid_files = 0
    total_files = 0
    
    for split in dirs:
        label_dir = os.path.join(base_path, split, "labels")
        if not os.path.exists(label_dir):
            print(f"Directory not found: {label_dir}")
            continue
            
        label_files = glob(os.path.join(label_dir, "*.txt"))
        print(f"Checking {len(label_files)} files in {label_dir}")
        
        #check each file with progress bar
        for file_path in tqdm(label_files, desc=f"Checking {split}"):
            total_files += 1
            invalid_reasons = check_polygon_file(file_path)
            
            if invalid_reasons:
                invalid_files += 1
                print(f"\nInvalid file: {file_path}")
                for reason in invalid_reasons:
                    print(f"  - {reason}")
    
    print(f"\nSummary: Found {invalid_files} invalid files out of {total_files} total files")

#check all labels
base_path = "datasets/BDD100K_Lane"
check_all_labels(base_path)