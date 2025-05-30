import os, torch
from data_preprocessing.dataset_loader import make_continuous_loader
import data_preprocessing.video_preprocess as preprocess
import LTC.model.model_visualisation as vis
from LTC.model.convLTC import ConvLTC
from LTC.model.neural_wiring import NeuralCircuitPolicy as ncp
from data_preprocessing.training_utilities import train_continuous_model
from data_preprocessing.model_utilities import plot_loss

#for reproducibility
seed = 24573471
torch.manual_seed(seed)

#----------directiories--------------
root_dir = r"datasets\Comma_ai_dataset" #for dataset
checkpoint_dir = r"LTC\checkpoint_weights" #for saving/storing checkpoint weights during training
metrics_path = r"LTC\training_metrics" #store the current training metrics (updated every save_freq (which is currently set to 10))
root_img_path = r"images" #storing configuration images

# os.makedirs(checkpoint_dir, exist_ok=True)
# os.makedirs(metrics_path, exist_ok=True)
#---------------------------------------

#---------------training hyper‑params------------------
seq_len = 24  #sequence length for each batch, longer sequences are better
seq_stride = 8  #stride between sequences for better continuity, i.e seg_1 = 0-23, then seg_2 = 15-33, maintains a balance of almost n% familiar frames and m% newer frames for training and balancing temporal continuity
batch_size = 32 #loads seq_len frames * batch_size into memory during training, reduce batch size or seq_len based on your device requirements
epochs = 170 #200
learning_rate = 0.0005
conv_grad_scaling = 1.0  #higher scaling means conv learns at a faster rate than LTC
curve_factor = 0.25  #for weighted loss for steering
accel_factor = 0.30  #higher factor for acceleration magnitude weighting
augment_prob = 0.35  #augmentation probability
quick_test = False  #set to False for full training
loss_type = "magnitude" #other options include 'sequence', or 'combined'

#use speed as additional feature
use_speed = True  #control flag for using speed data
speed_embedding_size = 8  #size matches the embedding in ConvLTC
#---------------------------------------

#-------------------Data loading------------------
#get the labels, steering angle and car_accel
labels = [preprocess.telemetry_keys[0], preprocess.telemetry_keys[4]] #steering angle, car_accel
print(f"using label(s): {labels[:]}")

#get speed feature for input (not prediction target)
speed_feature = ['speed_abs'] if use_speed else None
print(f"using speed feature as input: {speed_feature}")

#for putting data on same device as model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")

#this builds the memory‑efficient continuous dataloader
train_loader = make_continuous_loader(
    root_dir, #root directory of dataset containing splits
    split='train', #the split to use, can be 'val' or 'test'
    telemetry_keys=labels, #labels to predict
    seq_len=seq_len, #sequence length of each video segment in a batch
    seq_stride=seq_stride, #overlap amount for consecutive segments from a single video file
    batch_size=batch_size, #parallel processing
    augment_prob=augment_prob, #probability to apply random augmentation to video segment during input
    normalise=True, #normalise pixel values
    shuffle_files=True, #shuffle files randomly (not segments)
    use_test_slices=quick_test, #set to false for full training
    speed_feature=speed_feature, #use speed_abs as the speed feature to train on, not using negative speed bcause that indicates reversing, model is only training on front facing video streams
    excluded_files=None #automatically handled to exclude 1 of the training files internally so ill leave it as none here, exlcuded the 12GB file from training because it was too long
)

print(f"train loader created - using augmentation probability: {augment_prob}")
#---------------------------------------

#--------------Model Building------------------
#expected dimensions of frame
H, W = 160, 320 #frame height x width, this is based off the comma ai dataset only
print(f"frame height: {H}, frame width: {W}")

#this initialises ncp wiring, num_filters x features_per_filter + the feature size of the speed embedding inside the convLTC code is the number of sensory neurons (input layer)
num_filters = 8
features_per_filter = 6
num_sensory = num_filters * features_per_filter + speed_embedding_size

print(f"total sensory neurons: {num_sensory}")

#create the neural wiring of the LTC model
wire = ncp(
    inter_neurons=24,
    command_neurons=12,
    motor_neurons=len(labels), #the number of prediction targets
    outgoing_sensory_neurons=7, #must be less than or equal to number of interneurons
    outgoing_inter_neurons=5, #must be less than or equal to number of command neurons
    num_of_recurrent_connections=5,
    outgoing_command_neurons=4,  #must be less than or equal to number of command neurons
    seed=seed
)
wire.build(num_sensory) #initialises the model once input_dim is called during built()

#optional wiring visuals #uncomment if any modifications to the current configuration is made
# vis.plot_adjacency_matrices(wire, save_path=f"{root_img_path}\\adjacency_matrices.png")
# vis.view_neural_wiring(wire, save_path=f"{root_img_path}\\LTC_neural_structure.png", show=False)

#this constructs the model with the convolutional head
model = ConvLTC(
    wiring=wire, #the ncp object/ neural wiring of the model
    image_height=H, 
    image_width=W,
    num_filters=num_filters, #number of output filters in the 'dense' layer of the convolution; we use 8 which you can see in images\Convolution_Head-structure.png
    features_per_filter=features_per_filter, #number of features that each of the dense layer produces
    return_sequences=True, #returns prediction and hidden state history, can be set to false during inference
    input_mapping="affine", #input weight and bias mapping; otherwise use 'linear' for just input weight and no bias
    output_mapping="affine", #same applies for output
    ode_unfolds=6, #number of ODE approximations to perform in a single time step (recurrency to evolve the hidden state)
    implicit_constraints=False, #optionally enforces positive values for certain parameters, set to false because we apply it seperatly during training using apply_param_constraints (taken from WormNet Implementation)
    use_speed_input=use_speed  #pass the flag to the model
).to(device)

print("model constructed")
print(f"Number of total neurons: {model.ltc_cell.total_neurons}")
print(f"Number of internal neurons: {model.ltc_cell.internal_neuron_size}")

#count total parameters
total_params = sum(p.numel() for p in model.parameters())
conv_params = sum(p.numel() for name, p in model.named_parameters() if 'conv_head' in name)
speed_params = sum(p.numel() for name, p in model.named_parameters() if 'speed_embedding' in name) if use_speed else 0
ltc_params = total_params - conv_params - speed_params

print(f"total model parameters: {total_params:,}")
print(f"conv head parameters: {conv_params:,} ({conv_params/total_params*100:.1f}%)")

print(f"speed embedding parameters: {speed_params:,} ({speed_params/total_params*100:.1f}%)")
print(f"ltc parameters: {ltc_params:,} ({ltc_params/total_params*100:.1f}%)")
#---------------------------------------

#--------------Quick test training to ensure functions work------------------
# #run quick test to verify training functionality
# def test_training(model, root_dir, n_frames=100):
#     print("running quick test training to verify functionality...")
    
#     #create test dataloader with minimal data
#     test_loader = make_continuous_loader(
#         root_dir, 
#         split='train',
#         telemetry_keys=labels,
#         seq_len=seq_len,
#         batch_size=4,
#         augment_prob=0.0,
#         use_test_slices=True,
#         test_frames=n_frames,
#         speed_feature=speed_feature  #include speed feature in test
#     )
    
#     #run training for 3 epochs
#     test_history = train_continuous_model(
#         model=model,
#         train_loader=test_loader,
#         epochs=3,
#         device=device,
#         save_path=None,  #don't save during test
#         metrics_path=None,  #don't save metrics during test
#         learning_rate=learning_rate,
#         conv_grad_scaling=conv_grad_scaling,
#         curve_factor=curve_factor,
#         mag_factor=curve_factor, #steering factor
#         accel_factor=accel_factor, #acceleration factor
#         loss_type=loss_type,
#         telemetry_labels=labels
#     )
    
#     print("test training complete!")
#     print(f"test loss history: {test_history['weighted_loss']}")
    
#     return test_history

#uncomment to run quick test
# history = test_training(model, root_dir, n_frames=100)
#-----------------------------------------------------------------------------


#--------------full training------------------
#training function inherently saves all the training history data at every epoch, it gets overwritten to the final epoch for full training evaluation
#but keep intermediate epoch ones incase we do early stopping; view training_utilities.py for training functions and set up
history = train_continuous_model(
    model=model, #pass model
    train_loader=train_loader, #pass data
    epochs=epochs,
    device=device,
    save_path=checkpoint_dir, #set weight checkpoint saving directory
    metrics_path=metrics_path, #set training metrics directory
    save_freq=10,  #save checkpoint every 10 epochs
    learning_rate=learning_rate, #optimiser learning rate; by default we are using adam, view training_utilities.py
    conv_grad_scaling=conv_grad_scaling,
    curve_factor=curve_factor, #steering factor
    mag_factor=curve_factor, #steering factor
    accel_factor=accel_factor, #acceleration factor
    loss_type=loss_type,
    telemetry_labels=labels
)


print("training complete and metrics saved")

#plot results
plot_loss(metrics_path=metrics_path)