import torch
import torch.nn as nn
import numpy as np
from LTC.ltccell import LTCCell
#---------------------------
# Liquid Time-Constant Model - Develops full LTC model, any adjustments to layer wise architecture gets modified here; assumes batch first format for training
#---------------------------
class LTC(nn.Module):
    def __init__(self,
                 wiring, #neural circuit policy wiring object
                 return_sequences=True, #whether to return the prediction history as a list or not
                 input_mapping="affine", #input transformation type
                 output_mapping="affine", #output transformation type
                 ode_unfolds=6, #ode steps in single time step process
                 epsilon=1e-8, #avoid div by 0
                 implicit_constraints=False): #whether to use softplus for parameter constraints
        
        super(LTC, self).__init__() #inherent pytorch

        self.wiring = wiring
        self.return_sequences = return_sequences

        #any pre-processing layers get added here

        #initialise ltc cell with wiring
        self.ltc_cell = LTCCell(
            wiring=wiring,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            ode_unfolds=ode_unfolds,
            epsilon=epsilon,
            implicit_constraints=implicit_constraints,
        )

        #any post processing layers get added here 

    def forward(self, input_sequence, hidden_state=None, time_span=None):
        device = input_sequence.device
        #input_sequence: tensor with shape (batch_size, sequence_length, features) assuming batch first
        sequence = input_sequence
        batch_size, seq_len = input_sequence.size(0), input_sequence.size(1) #assuming expected shape

        if hidden_state is None:
            hidden_state = torch.zeros((batch_size, self.ltc_cell.internal_neuron_size), device=device)

        output_sequences = [] #store regression outputs which should be same length as seq_len

        for t in range(seq_len):
            time_delta = 1.0 if time_span is None else time_span[:, t] #otherwise assuming time_span input is same as sequence length with different time values for more realisitc time dynamics
            x = sequence[:, t]
            output, hidden_state = self.ltc_cell(x, hidden_state, time_delta)

            if self.return_sequences:
                output_sequences.append(output) 

        if self.return_sequences:
            readout = torch.stack(output_sequences, dim=1)
        else:
            readout = output
        
        return readout, hidden_state
        

        


