import yolo_lane_seg_training as tc #as 'training code' (tc)
import matplotlib.pyplot as plt

#test results on lane
best_model_path = r"Model_dev\yolo_model_weights\training_results\bdd100k_lane_training\weights\best.pt" #nano
# best_model_path = r"Model_dev\yolo_model_weights\training_results\bdd100k_lane_training_yolov11s-seg\weights\best.pt"

# test_sample_path = r"datasets\BDD100K_Lane\test\images\fde816b0-7d366afb.jpg"
# test_sample_path = r"datasets\BDD100K_Env\train\images\0a172b0e-136b7a63.jpg"
# test_sample_path = r"datasets\BDD100K_Env\train\images\b090bc5d-79803e42.jpg"
# test_sample_path = r"datasets\BDD100K_Env\train\images\b0e2355d-5d91d031.jpg"

if __name__ == '__main__':
    tc.multiprocessing.freeze_support()
    model = tc.YOLO(best_model_path)
    test_results = model.val(data=tc.lane_yaml_path, split='test')
    # results = model(test_sample_path)

    # plt.figure(figsize=(10, 10))
    # plt.imshow(results[0].plot())
    # plt.axis('off')
    # plt.show()