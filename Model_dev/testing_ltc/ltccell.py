import torch
import torch.nn as nn
import numpy as np 
import torch.optim as optim 
import wiring_class as ncp

#---------------------------
# Liquid Time-Constant Cell
#---------------------------
class LTCCell(nn.Module):
    def __init__(
        self, 
        wiring, 
        in_features=None, 
        input_mapping="affine", #or linear; both mappings do the same thing but affine allows enables a bias parameter to be part of the network
        output_mapping="affine", 
        ode_unfolds=6, 
        epsilon=1e-8,
        implicit_constraints=False):

        super(LTCCell, self).__init__() #inherent from pytorch
        
        #initialise wiring if input features provided
        if in_features is not None:
            wiring.build(in_features) 
        
        if not wiring.is_initialised():
            raise ValueError(f"error initialising ncp neuron wiring, please provide in_features or call wiring.build() first")
        
        #function to enforce positive parameter values
        #softplus is a similar activation function to ReLU() with a learnable parameter beta (if trainable = True, it is False by default), the only difference is it doesn't clip values at 0 but makes a smoother transition around 0
        #nn.Identity applies no transformation to the input, it simply does nothing and can be a placeholder for skip connections without impacting the entire network structure
        self.make_positive = nn.Softplus() if implicit_constraints else nn.Identity()
        self.implicit_constraints = implicit_constraints
        
        #parameter initialisation ranges, taken from Official github
        self.init_ranges = {
            "leakage_conductance": (0.001, 1.0), #controls leak current
            "reverse_potential": (-0.2, 0.2),   
            "membrane_capacitance": (0.4, 0.6),  #time constant scaling
            "w": (0.001, 1.0), #synaptic weights
            "sigma": (3, 8),  
            "mu": (0.3, 0.8),     
            "sensory_w": (0.001, 1.0),     
            "sensory_sigma": (3, 8),              
            "sensory_mu": (0.3, 0.8)           
        }
        
        #store configuration
        self.wiring = wiring
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.ode_unfolds = ode_unfolds  #how many times to approximate hidden state evolution per single time step process, by default every time step has 6 unfolds
        self.epsilon = epsilon  #small constant to prevent division by zero
        self.clip = nn.ReLU()  #used to enforce positive values if not using implicit constraints
        
        #initialise all trainable parameters
        self.initialise_parameters()
    #internal state = number of total neurons in cell
    @property
    def state_size(self):
        return self.wiring.total_neurons
    
    #input dimensions of cell
    @property
    def sensory_size(self):
        return self.wiring.input_dim
    
    #output dimensions either for predictive output or hidden state
    @property
    def motor_size(self):
        return self.wiring.output_dim
    
    #encapsulate motor_size function
    @property
    def output_size(self):
        return self.motor_size
    
    #neuron to neuron connections
    @property
    def synapse_count(self):
        return np.sum(np.abs(self.wiring.neuron_connections))
    
    #sensory neuron connections
    @property
    def sensory_synapse_count(self):
        return np.sum(np.abs(self.wiring.sensory_connections))
    
    #---------------------------neurons/parameter initialisation---------------------------
    def add_weight(self, name, init_value, requires_grad=True):
        param = nn.Parameter(init_value, requires_grad=requires_grad)
        self.register_parameter(name, param) #store the parameter with a name to keep track i.e sensory weights
        return param
    
    #used to initialise trainable parameters with min max values from init_ranges dictionary
    def get_init_value(self, shape, param_name):
        minval, maxval = self.init_ranges[param_name] 

        if minval == maxval:
            return torch.ones(shape) * minval 
        else:
            return torch.rand(*shape) * (maxval - minval) + minval #*shape passes shape iterable and unpacks every element

    #all trainable parameters of LTCCell   
    def initialise_parameters(self):
        self.params = {}

        #neuron parameters
        keys = ["leakage_conductance", "reverse_potential", "membrane_capacitance"]
        for name in keys:
            self.params[name] = self.add_weight(name=name, init_value=self.get_init_value((self.state_size,), name))
        
        #neuron to neuron connection parameters
        keys = ["w", "sigma", "mu"]
        for name in keys:
            self.params[name] = self.add_weight(name=name, init_value=self.get_init_value((self.state_size, self.state_size), name))
        
        #neuron to neuron reverse potential
        self.params['ntn_reverse_potential'] = self.add_weight(name="ntn_reverse_potential", init_value=torch.Tensor(self.wiring.neuron_reverse_potential_initialiser()))

        #sensory to neuron connection parameters
        keys = ["sensory_w", "sensory_sigma", "sensory_mu"]
        for name in keys:
            self.params[name] = self.add_weight(name=name, init_value=self.get_init_value((self.sensory_size, self.state_size), name))
        
        #sensory reverse potential
        self.params["sensory_reverse_potential"] = self.add_weight(name="sensory_reverse_potential", init_value=torch.Tensor(self.wiring.input_reverse_potential_initialiser()))

        #sparsity masks, they are non-trainable
        keys = ["sparsity_mask", "sensory_sparsity_mask"]
        self.params[keys[0]] = self.add_weight(name=keys[0], init_value=torch.Tensor(np.abs(self.wiring.neuron_connections)), requires_grad=False)
        self.params[keys[1]] = self.add_weight(name=keys[1], init_value=torch.Tensor(np.abs(self.wiring.sensory_connections)), requires_grad=False)

        #optional input and output mappings
        if self.input_mapping in ["affine", "linear"]:
            self.params["input_weights"] = self.add_weight(name="input_weights", init_value=torch.ones((self.sensory_size,)))

        if self.input_mapping == "affine":
            self.params["input_bias"] = self.add_weight(name="input_bias", init_value=torch.zeros((self.sensory_size,)))
        
        if self.output_mapping in ["affine", "linear"]:
            self.params["output_weights"] = self.add_weight(name="output_weights", init_value=torch.ones((self.motor_size,)))

        if self.output_mapping == "affine":
            self.params["output_bias"] = self.add_weight(name="output_bias", init_value=torch.zeros((self.motor_size,)))

    #this may be optional, might remove if not necessary
    def apply_constraints(self):
        #enforce constraints on parameters if not using implicit constraints
        if not self.implicit_constraints:
            #ensure positive values for parameters that must be positive
            self.params["w"].data = self.clip(self.params["w"].data)
            self.params["sensory_w"].data = self.clip(self.params["sensory_w"].data)
            self.params["membrane_capacitance"].data = self.clip(self.params["membrane_capacitance"].data)
            self.params["leakage_conductance"].data = self.clip(self.params["leakage_conductance"].data)

    #---------------------------forward pass for single time step---------------------------
    #forward pass: input -> map_input -> sensory gate -> fused solver -> map outputs
    def forward(self, x, state, elapsed_time=1.0):
        x_transformed = self.map_input(x)

        next_state = self.fused_solver(x_transformed, state, elapsed_time)

        outputs = self.map_outputs(next_state)

        return outputs, next_state 
    
    def map_input(self, x):
        if self.input_mapping in ["affine", 'linear']: #both affine and linear perform linear transformation, affine provides the additional bias parameter calculation
            x = x * self.params['input_weights']
        
        if self.input_mapping == "affine":
            x = x + self.params['input_bias']
        
        return x

    def map_outputs(self, state):
        output = state 
        if self.motor_size < self.state_size:
            output = output[:, 0:self.motor_size] #output is sliced to motor amount of neurons
        
        if self.output_mapping in ["affine", 'linear']:
            output = output * self.params["output_weights"]
        if self.output_mapping == "affine":
            output = output + self.params['output_bias']
        
        return output
    
    def sigmoid_gate(self, voltage, mu, sigma):
        previous_voltage = torch.unsqueeze(voltage, -1)
        mu = previous_voltage - mu
        x = sigma * mu 
        return torch.sigmoid(x)
    
    def fused_solver(self, x_transformed, state, elapsed_time):
        previous_voltage = state #hidden state
        
        #precompute sensory effects here
        sensory_w_activation = self.make_positive(self.params["sensory_w"]) * self.sigmoid_gate(x_transformed, self.params["sensory_mu"], self.params["sensory_sigma"])
        sensory_w_activation = sensory_w_activation * self.params["sensory_sparsity_mask"]

        sensory_reverse_activation = sensory_w_activation * self.params["sensory_reverse_potential"]

        weight_sensory_numerator = torch.sum(sensory_reverse_activation, dim=1) #reverse potential as numerator
        weight_sensory_denominator = torch.sum(sensory_w_activation, dim=1) #sensory weights as denom

        scaled_capacitance = self.make_positive(self.params["membrane_capacitance"]) / (elapsed_time / self.ode_unfolds)

        #unfold the ODE multiple times into one RNN step
        connection_weights = self.make_positive(self.params["w"])
        for t in range(self.ode_unfolds):
            weight_activation = connection_weights * self.sigmoid_gate(previous_voltage, self.params["mu"], self.params["sigma"])
            weight_activation = weight_activation * self.params['sparsity_mask']
            reverse_activation = weight_activation * self.params['ntn_reverse_potential']

            weight_numerator = torch.sum(reverse_activation, dim=1) + weight_sensory_numerator
            weight_denominator = torch.sum(weight_activation, dim=1) + weight_sensory_denominator

            leakage_conductance = self.make_positive(self.params["leakage_conductance"])

            numerator = scaled_capacitance * previous_voltage + leakage_conductance * self.params["reverse_potential"] + weight_numerator 
            denominator = scaled_capacitance + leakage_conductance + weight_denominator

            previous_voltage = numerator / (denominator + self.epsilon)
        
        return previous_voltage #return evolved hidden state
    
