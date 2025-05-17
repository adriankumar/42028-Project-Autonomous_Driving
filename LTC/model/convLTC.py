import torch
import torch.nn as nn
from LTC.model.ltccell import LTCCell
from LTC.model.conv_head import ConvHead

#---------------------------
# Conv Liquid Time-Constant Model - Integrates convolutional feature extraction with LTC dynamics - this is the model we use
#---------------------------
class ConvLTC(nn.Module):
    def __init__(self,
                 wiring, #neural circuit policy wiring object
                 image_height=160, #image height for input
                 image_width=320, #image width for input
                 num_filters=8, #number of convolutional filters
                 features_per_filter=8, #features per filter for conv head
                 return_sequences=True, #whether to return the prediction and hidden state history, otherwise returns most recent prediction and current hidden state
                 input_mapping="affine", #input transformation type
                 output_mapping="affine", #output transformation type
                 ode_unfolds=6, #ode steps in single time step process
                 epsilon=1e-8, #avoid div by 0
                 implicit_constraints=False, #whether to use softplus for parameter constraints, otherwise, parameter constraints are applied during training
                 use_speed_input=True): #using speed as a feature input; controlled flag to append speed data into ltccell
        
        super(ConvLTC, self).__init__() #inherent pytorch

        self.wiring = wiring
        self.return_sequences = return_sequences
        self.image_height = image_height
        self.image_width = image_width
        self.use_speed = use_speed_input

        if self.use_speed:
            # self.speed_embedding = nn.Linear(1, 4) #scalar speed value embedded into 4 dim feature space <-- old version
            #new speed embedding 
            self.speed_embedding = nn.Sequential(
                nn.Linear(1, 16), #scalar speed value expanded to 16 features
                nn.ReLU(), #non-linear activation
                nn.Dropout(0.2), #prevent full dense layer
                nn.Linear(16, 8) #8 feature dim for speed
            )

        #create conv head + LTC (NCP)
        self.initialise_network(
            wiring=wiring,
            num_filters=num_filters,
            features_per_filter=features_per_filter,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            ode_unfolds=ode_unfolds,
            epsilon=epsilon,
            implicit_constraints=implicit_constraints
        )
        
    def initialise_network(self, wiring, num_filters, features_per_filter, input_mapping, output_mapping, ode_unfolds, epsilon, implicit_constraints):

        #create convolutional head for feature extraction
        self.conv_head = ConvHead(num_filters=num_filters, 
                                  features_per_filter=features_per_filter,
                                  img_h=self.image_height,
                                  img_w=self.image_width,
                                  channels=3) #assuming all video files are converted into 3 channels before input
        
        #initialise ltc cell with wiring
        self.ltc_cell = LTCCell(wiring=wiring,
                                input_mapping=input_mapping,
                                output_mapping=output_mapping,
                                ode_unfolds=ode_unfolds,
                                epsilon=epsilon,
                                implicit_constraints=implicit_constraints)

    #feature extract using conv, recurrent process via ltccell    
    def forward(self, input_sequence, speed_sequence=None, hidden_state=None, time_span=None):
        device = input_sequence.device
        
        #extract features from images
        #input_sequence should have shape [batch_size, seq_len, height, width, channels]
        batch_size, seq_len = input_sequence.size(0), input_sequence.size(1)

        #input to conv layer, conv head already handles input reshaping
        features = self.conv_head(input_sequence) #shape is seq x batch x features for ltc cell input - time major format
        features = features / (features.std() + 1e-5)  #normalise features
        
        #initialise hidden state if not provided
        if hidden_state is None:
            hidden_state = torch.zeros((batch_size, self.ltc_cell.internal_neuron_size), device=device) #hidden state remains as shape batch x  

        output_sequences = [] #store regression outputs
        all_hidden_states = [] #store all hidden states for analysis

        #process sequence through LTC cell
        for t in range(seq_len):

            time_delta = 1.0 if time_span is None else time_span[:, t] #if deciding to use actual time data from input, otherwise delta is by 1 second
            x = features[t] #sampling from time step t, x is now shape batch x feature dim

            #handling new speed feature input
            if self.use_speed and speed_sequence is not None:
                speed_t = speed_sequence[:, t, 0].unsqueeze(-1) #is currently in shape batch x seq=1 x 1, reshape to batch x 1 
                speed_embedded = self.speed_embedding(speed_t) #embedding layer for speed features

                # speed_embedded = speed_embedded / (speed_embedded.std() + 1e-5) #normalise --old normalisation, removed, was causing terrible prediction results in car accel

                speed_embedded = torch.tanh(speed_embedded) #alternative normalisation to preserve relative speed magntiudes
                x = torch.cat([x, speed_embedded], dim=1) #append speed feature to visual features, new input dim is now (num_filters * features_per_filter) + output_dim_of_speed_embedding

            #forward pass everything into ltccell for recurrent processing
            output, hidden_state = self.ltc_cell(x, hidden_state, time_delta) #output has shape batch x motor_size, hidden state has batch x internal_neuron_size

            all_hidden_states.append(hidden_state)

            if self.return_sequences: #store prediction output sequence
                output_sequences.append(output) 

        #stack in time major format
        if self.return_sequences:
            readout = torch.stack(output_sequences, dim=0) #seq x batch x motor_size
            all_hidden_states = torch.stack(all_hidden_states, dim=0) #seq x batch x internal_neuron_size

            #convert back to batch x seq x motor size for compatability with rest of code
            readout = readout.permute(1, 0, 2) #batch x seq x outputdim/motor size
            all_hidden_states = all_hidden_states.permute(1, 0, 2) #batch x seq x internal neuron size

        else:
            readout = output #final output, batch x motor_size
            all_hidden_states = hidden_state #batch x internal_neuron_size
        
        return readout, hidden_state, all_hidden_states #predictions, current hidden state, history of hidden state evolution
    


