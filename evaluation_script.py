import torch
import data_preprocessing.video_preprocess as preprocess
from data_preprocessing.model_utilities import plot_predictions, load_model, plot_hidden_state_activity, visualbackprop, plot_saliency_maps, plot_detailed_metrics, plot_loss
import LTC.model.model_visualisation as vis


#define telemetry labels and speed feature
labels = [preprocess.telemetry_keys[0], preprocess.telemetry_keys[4]] #steering_angle, car_accel
speed_feature = 'speed_abs'  #using speed_abs as our speed feature
use_speed = True  #control flag for using speed data

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_path = r"datasets\Comma_ai_dataset\train"
# val_path = r"datasets\Comma_ai_dataset\val"
test_path = r"datasets\Comma_ai_dataset\test"


model_path = r"LTC\checkpoint_weights\model_epoch170.pt"
root_img_path = r"model_images"

metrics_path = r"LTC\training_metrics"
metrics_file = r"e170_training_loss.npz"
metadata_file = r"e170_training_metadata.json"

#load model
model = load_model(model_path, device=device)
# vis.plot_adjacency_matrices(model.wiring, save_path=f"{root_img_path}\\adjacency_matrices.png")
# vis.view_neural_wiring(model.wiring, save_path=f"{root_img_path}\\LTC_neural_structure.png", show=False)

# print("\nplotting training metrics...")
# plot_loss(metrics_path, metrics_file)
# plot_detailed_metrics(metrics_path, metrics_file, metadata_file)

print("preparing for model evaluation")

#training data eval
train_0_test, labels_0_test, speed_0_test = preprocess.load_sample(
    datasplit_path=train_path,
    file_name=preprocess.train_files[0],
    telemetry_labels=labels,
    normalise=True,
    start=5260,
    end=5500,
    extract_speed_feature=True
)

# test_0_test, labels_0_test, speed_0_test = preprocess.load_sample(
#     datasplit_path=test_path,
#     file_name=preprocess.test_files[0], #smallest one
#     telemetry_labels=labels,
#     normalise=True,
#     start=5770,
#     end=6000,
#     extract_speed_feature=True
# )

#frame-by-frame inference with speed data
predictions, hidden_states = preprocess.process_video_inference(
    model=model,
    video_data=train_0_test, #replace with either train_0_test or test_0_test
    device=device,
    speed_data=speed_0_test,
    seq_len=1,  #process one frame at a time
    normalise=False  #already normalised in load_sample
)

#plot predictions vs ground truth
#plot all outputs
plot_predictions(predictions, labels_0_test)
#plot individual outputs
# plot_predictions(predictions, labels_0_test, output_idx=0, output_name="steering_angle")
# plot_predictions(predictions, labels_0_test, output_idx=1, output_name="car_accel")

#plot hidden state activity
# plot_hidden_state_activity(hidden_states, "ltc hidden state activity")

# #visualise saliency map for a subset of frames
# tensor_test = torch.tensor(test_0_test[700:705], dtype=torch.float32, device=device)
# tensor_test = tensor_test.permute(0, 2, 3, 1).unsqueeze(0) #[1, seq, H, W, C]

#saliency via visual backprop
#generate saliency maps with the proper input shape
# saliency_maps = visualbackprop(model, tensor_test, device=device)

#plot the saliency maps
# plot_saliency_maps(tensor_test, saliency_maps, alpha=1.0, save_path=f"{root_img_path}\\saliency_map_visual_backprop.png")