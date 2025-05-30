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
val_path = r"datasets\Comma_ai_dataset\val"
test_path = r"datasets\Comma_ai_dataset\test"


model_path = r"LTC\checkpoint_weights\model_epoch170.pt"
root_img_path = r"images"

metrics_path = r"LTC\training_metrics"
metrics_file = r"e170_training_loss.npz"
metadata_file = r"e170_training_metadata.json"

#load model
model = load_model(model_path, device=device)
# vis.plot_adjacency_matrices(model.wiring, save_path=f"{root_img_path}\\adjacency_matrices.png")
# vis.view_neural_wiring(model.wiring, save_path=f"{root_img_path}\\LTC_neural_structure_transparent.png", show=False)

print("\nplotting training metrics...")
plot_loss(metrics_path, metrics_file)
plot_detailed_metrics(metrics_path, metrics_file, metadata_file)

print("preparing for model evaluation")

# #load validation/test files
def load_segment(start, end, split_path, filename):
    video, tel_labels, speed = preprocess.load_sample(
        datasplit_path=split_path,
        file_name=filename,
        telemetry_labels=labels,
        normalise=True,
        start=start,
        end=end,
        extract_speed_feature=True
    )

    return video, tel_labels, speed 

# #process inference
def evaluate_segment(start, end, split_path, filename, seq_len=1, plot_hidden_states=False):

    video, tel_labels, speed = load_segment(start, end, split_path, filename) #load segment

    predictions, hidden_states = preprocess.process_video_inference(
        model=model,
        video_data=video,
        device=device,
        speed_data=speed,
        seq_len=seq_len,
        normalise=False
    )
    plot_predictions(predictions, tel_labels, output_idx=0, output_name="Steering Angles; True vs Predicted") #plot steering angle predictions
    plot_predictions(predictions, tel_labels, output_idx=1, output_name="Acceleration; True vs Predicted") #plot acceleration predictions

    if plot_hidden_states:
        plot_hidden_state_activity(hidden_states, "LTC hidden state activity over segment")

# #validation files-------------------
# #video 1--------------
# #seg 1 
evaluate_segment(7321, 9830, val_path, preprocess.val_files[0]) 

# #seg 2 
evaluate_segment(11295, 12000, val_path, preprocess.val_files[0]) 

# #seg 3 
evaluate_segment(15530, 17439, val_path, preprocess.val_files[0]) 
# #-------------

# #video 2--------------
# #seg 1 
evaluate_segment(4559, 7673, val_path, preprocess.val_files[1]) 

# #seg 2 
evaluate_segment(9250, 16475, val_path, preprocess.val_files[1]) 

# #seg 3 
evaluate_segment(39450, 42023, val_path, preprocess.val_files[1]) 
# #-------------


# #testing files-------------------
# #video 1--------------
# #seg 1 
evaluate_segment(4578, 9443, test_path, preprocess.test_files[0]) 

# #seg 2 
evaluate_segment(10118, 11100, test_path, preprocess.test_files[0]) 

# #seg 3 
evaluate_segment(12990, 15743, test_path, preprocess.test_files[0]) 
# #-------------

# #video 2--------------
# #seg 1 
evaluate_segment(8231, 17525, test_path, preprocess.test_files[1]) 

# #seg 2 
evaluate_segment(36555, 39830, test_path, preprocess.test_files[1]) 

# #seg 3 
evaluate_segment(46335, 48548, test_path, preprocess.test_files[1]) 
# #-------------

# #video 3--------------
# #seg 1 
evaluate_segment(4697, 5505, test_path, preprocess.test_files[2]) 

# #seg 2 
evaluate_segment(7526, 8435, test_path, preprocess.test_files[2]) 

# #seg 3 
evaluate_segment(14800, 41618, test_path, preprocess.test_files[2]) 
#-------------

# #visualise saliency map for a subset of frames
# random_sample, _, _ = load_segment(4000, 5000, test_path, preprocess.test_files[0])
# tensor_test = torch.tensor(random_sample[700:705], dtype=torch.float32, device=device)
# tensor_test = tensor_test.permute(0, 2, 3, 1).unsqueeze(0) #[1, seq, H, W, C]

#saliency via visual backprop
#generate saliency maps with the proper input shape
# saliency_maps = visualbackprop(model, tensor_test, device=device)

#plot the saliency maps
# plot_saliency_maps(tensor_test, saliency_maps, alpha=1.0, save_path=f"{root_img_path}\\saliency_map_visual_backprop.png")