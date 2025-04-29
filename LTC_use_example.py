import torch
from LTC.ltc_model import LTC 
from LTC.neural_wiring import NeuralCircuitPolicy as ncp
from LTC.model_visualisation import print_matrix_connections, view_neural_wiring, plot_adjacency_matrices

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 24573471

input_dim = 20 #same as number of sensory neurons
outgoing_sensory = 5 #must be less than or equal to number of interneurons

num_inter = 16
outgoing_inter = 3 #must be less than or equal to number of command neurons

num_command = 6
num_reccurrent = 2 
outgoing_command = 5

output_dim = 1 #same as number of motor neurons

#initialise wire object
wire = ncp(
    inter_neurons= num_inter,
    command_neurons= num_command,
    motor_neurons= output_dim,
    outgoing_sensory_neurons= outgoing_sensory,
    outgoing_inter_neurons= outgoing_inter,
    num_of_recurrent_connections= num_reccurrent,
    outgoing_command_neurons= outgoing_command,
    seed=seed
)

wire.build(input_dim) #initialise connectivity (adjacency matrices)

model = LTC(
    wiring=wire,
    return_sequences=True,
    input_mapping="affine",
    output_mapping="affine",
    ode_unfolds=6,
    implicit_constraints=True,
    device=device
)

# print_matrix_connections(wire)
plot_adjacency_matrices(wire, save_path="adjacency_matrices.png") #will save as image instead of popping up as window
view_neural_wiring(wire, save_path="LTC_neural_structure.png", show=False) #will save as image instead of popping up as window
