import os 
import multiprocessing
from ultralytics import YOLO 

#./ tells the system to work in the current directory, assuming you've opened vscode in '42028-Project-Autonomous_Driving' directory, then this path is in the 42028-Project-Autonomous_Driving/<folder> directory
#if you've opened vscode inside a diff folder then this path will not work
dataset_path = os.path.join(".\\", "datasets") 

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #suppress warning, need to potentially fix

yolo_weights_path = os.path.join(".\\", "Model_dev" ,"yolo_model_weights", "preset weights")

model_variants = [model for model in os.listdir(yolo_weights_path)] #present weight models use index 2 for smallest 'nano'

full_model_path = os.path.join(yolo_weights_path, model_variants[-1]) #model path

#we do not need to change the dataset paths inside the yaml because ultralytics has already set the default path to the dataset as 42028-Project-Autonomous_Driving/datasets, you can modify this in settings.json in ultralytics
lane_yaml_path = r"Model_dev\model_building\bdd100k_lane.yaml"
env_yaml_path = r"Model_dev\model_building\bdd100k_env.yaml"


train_params = {
    'data': lane_yaml_path,
    'epochs': 6, #training iter
    'batch': 8, #batch size, GPU dependent, if on cpu use batch size of 1-4
    'imgsz': 640, #image resize, 640 is standard, 320 is faster, 1280 is high quality, must be divisble by 32
    # 'patience': 20, #patience is how many epochs do you want to train on out of the full epoch length for early stopping, i.e training stops at 20 iterations even if epochs = > 20
    'project': os.path.join("Model_dev", "yolo_model_weights", "training_results"),  #storing results
    'name': 'bdd100k_lane_training_yolov11s-seg', #name of model being trained, ultralytics automatically appends an index if you use the same name for multiple training attempts
    'optimizer': 'AdamW', # 'SGD', 'Adam', 'AdamW'
    'lr0': 0.001,
    'lrf': 0.01,
    'dropout': 0.2,
    'mask_ratio': 4, #mask downsampling ratio
    'overlap_mask': True,
    'device': 0, #GPU, use 'cpu' if no cuda access
    'val': True,
    }
    
if __name__ == '__main__':
#windows requires this for multiprocessing
    multiprocessing.freeze_support()

    #https://docs.ultralytics.com/usage/cfg/#__tabbed_1_2 <-- training stuff
    model = YOLO(full_model_path) #yolov11n-seg if index is 2, yolov11s-seg if index is -1 (last)

    results = model.train(**train_params) 
    
    print(f"model results saved to: {train_params['project']}/{train_params['name']}")
    print(f"model best weights saved to: {train_params['project']}/{train_params['name']}/weights/best.pt")