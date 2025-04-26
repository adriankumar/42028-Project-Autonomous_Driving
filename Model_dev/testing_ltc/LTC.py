import torch
import torch.nn as nn
import numpy as np
from ltccell import LTCCell
import wiring_class as ncp

#---------------------------
# Liquid Time-Constant Model
#---------------------------
class LTC(nn.Module):
    def __init__(
        self,
        input_size, #input features
        wiring, #neural circuit policy wiring object
        return_sequences=True, #whether to return all time steps or just final output
        batch_first=True, #whether batch is first dimension (batch, time, features) or (time, batch, features)
        input_mapping="affine", #input transformation type
        output_mapping="affine", #output transformation type
        ode_unfolds=6, #ode steps in single time step process
        epsilon=1e-8, #avoid div by 0
        implicit_constraints=False,): #whether to use softplus for parameter constraints


        super(LTC, self).__init__() #inherent pytorch
        self.input_size = input_size
        self.wiring = wiring
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        
        #initialise ltc cell with wiring
        self.rnn_cell = LTCCell(
            wiring=wiring,
            in_features=input_size,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            ode_unfolds=ode_unfolds,
            epsilon=epsilon,
            implicit_constraints=implicit_constraints
        )
    
    @property
    def state_size(self):
        return self.wiring.total_neurons
    
    @property
    def sensory_size(self):
        return self.wiring.input_dim
    
    @property
    def motor_size(self):
        return self.wiring.output_dim
    
    @property
    def output_size(self):
        return self.motor_size
    
    @property
    def synapse_count(self):
        return np.sum(np.abs(self.wiring.neuron_connections))
    
    @property
    def sensory_synapse_count(self):
        return np.sum(np.abs(self.wiring.sensory_connections))
    
    def forward(self, input_sequence, hidden_state=None, timespans=None):
        #process input sequence through ltc network
        #input_sequence: tensor with shape (batch_size, sequence_length, features) if batch_first=True,
        #or (sequence_length, batch_size, features) if batch_first=False
        #hidden_state: initial hidden state, zeros used if None
        #timespans: optional tensor of time intervals between steps
        #returns: output (full sequence or last timestep) and final hidden state

        device = input_sequence.device #assuming input sets the device gpu or cpu
        is_batched = input_sequence.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0
        
        #handle non-batched input (add batch dimension)
        if not is_batched:
            input_sequence = input_sequence.unsqueeze(batch_dim)
            if timespans is not None:
                timespans = timespans.unsqueeze(batch_dim)
        
        #get batch size and sequence length
        batch_size, seq_len = input_sequence.size(batch_dim), input_sequence.size(seq_dim)
        
        #initialise hidden state if not provided
        if hidden_state is None:
            h_state = torch.zeros((batch_size, self.state_size), device=device)
        else:
            h_state = hidden_state
            
            #handle shape validation
            if is_batched:
                if h_state.dim() != 2:
                    raise RuntimeError(f"for batched 2-d input, h_state should also be 2-d but got {h_state.dim()}-d tensor")
            else:
                #non-batched mode
                if h_state.dim() != 1:
                    raise RuntimeError(f"for unbatched 1-d input, h_state should also be 1-d but got {h_state.dim()}-d tensor")
                h_state = h_state.unsqueeze(0)
        
        #process sequence step by step
        output_sequence = []
        for t in range(seq_len):
            #get inputs for current timestep
            if self.batch_first:
                inputs = input_sequence[:, t]
                time_delta = 1.0 if timespans is None else timespans[:, t].squeeze()
            else:
                inputs = input_sequence[t]
                time_delta = 1.0 if timespans is None else timespans[t].squeeze()
            
            #apply ltc cell
            h_out, h_state = self.rnn_cell(inputs, h_state, time_delta)
            
            #collect outputs if returning full sequence
            if self.return_sequences:
                output_sequence.append(h_out)
        
        #prepare output based on return_sequences flag
        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            readout = torch.stack(output_sequence, dim=stack_dim)
        else:
            readout = h_out
        
        #handle non-batched output
        if not is_batched:
            readout = readout.squeeze(batch_dim)
            h_state = h_state[0]
        
        return readout, h_state