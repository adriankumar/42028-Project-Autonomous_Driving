from ultralytics import YOLO 
import os
import multiprocessing

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #suppress warning, need to potentially fix

#note that we are overriding the 80 original classes with strictly our lane segmentation

#.pt loads models with offical pre-trained weights
n_seg = "yolo_models/yolo11n-seg.pt"
s_seg = "yolo_models/yolo11s-seg.pt"
m_seg = "yolo_models/yolo11m-seg.pt"
l_seg = "yolo_models/yolo11l-seg.pt"

#yolov11-seg model variants
model_variant = [n_seg, s_seg, m_seg, l_seg]


if __name__ == '__main__':
    #windows requires this for multiprocessing
    multiprocessing.freeze_support()
    
    #model setup
    model = YOLO(model_variant[0])
    
    dataset_yaml_path = os.path.join("datasets", "BDD100K_Lane", "bdd100k_lane.yaml")
    
    #training parameters, research more and add more or change the current ones to validate performance
    train_params = {
        'data': dataset_yaml_path,
        'epochs': 2,
        'batch': 8,
        'imgsz': 640,
        'patience': 20,
        'project': os.path.join("yolo_models", "yolo_training_results"),
        'name': 'bdd100k_lane_training6',
        'optimizer': 'AdamW',
        'mask_ratio': 4,
        'overlap_mask': True,
        'device': 0,
        # 'workers': 2,
        'deterministic': True,
    }
    
    #train model
    results = model.train(**train_params)
    
    print(f"model results saved to: {train_params['project']}/{train_params['name']}")
    print(f"model best weights saved to: {train_params['project']}/{train_params['name']}/weights/best.pt")