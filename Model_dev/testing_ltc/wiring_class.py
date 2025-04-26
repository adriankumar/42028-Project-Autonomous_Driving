import numpy as np 

#the wiring class defines the neural connectivity within each LTCCell; dense connectivity, sparse connectivity, and neural circuit policy, for simplicity we only implement NCP as it provides specific neurons
#in the wiring and their connections such as:
#sensory -> External inputs (video frames)
#inter -> First processing layer (basic visual features)
#command -> Middle layer with recurrent connections (context/memory)
#motor -> Output layer (steering angles)

#---------------------------
# Base neuron wiring class
#---------------------------
class NeuronWiring:
    def __init__(self, total_neurons):
        self.total_neurons = total_neurons
        self.neuron_connections = np.zeros([total_neurons, total_neurons], dtype=np.int8)  #adjacency matrix defining connections between neurons (square matrix n x n); using np.int8 for 8-bit integer for memory efficiency instead of original np.int32 ((-1, 0, 1))
        self.sensory_connections = None #adjacency matrix defining connections between input/sensory neurons
        self.input_dim = None #number of input features from external time series
        self.output_dim = None #number of output features (motor neurons)

    #python decorator for simplified calling of expressions and variables
    #can be accessed without parentheses (wiring.layer_count instead of wiring.layer_count())
    #can be computed on-demand rather than stored in memory; improved computational efficiency
    #allows controlled access (read-only by default)
    @property 
    def layer_count(self):
        return 1 #default number of layers, this will change
    
    def get_neurons_of_layer(self): #i.e with sensory, inter, command, motor, we would have 4 layers each with their own set of neurons
        return list(range(self.total_neurons))

    def is_initialised(self): #is the neuron wiring initialised?
        return self.input_dim is not None
    
    def set_input_dim(self, input_dim): #defines expected input size and creates the sensory adjacency matrix
        self.input_dim = input_dim
        self.sensory_connections = np.zeros([input_dim, self.total_neurons], dtype=np.int8) #creates connection mapping (adjacency matrix) from input to sensory neurons

    #number of output features
    def set_output_dim(self, output_dim):
        self.output_dim = output_dim
    
    def build(self, input_dim): 
        #error handling of conflicting input size
        if self.input_dim is not None and self.input_dim != input_dim:
            raise ValueError(f"Conflicting input dimensions; input dim is: {self.input_dim}, but read {input_dim}")
        
        if self.input_dim is None:
            self.set_input_dim(input_dim)
    
    #since original code utilises copy(neuron_connections) with orginal dtype=np.int32, the two code lines were simply np.copy(neuron_connections) & np.copy(sensory_connections)
    #but we changed the dtype to np.int8 since those adjacecy matrices only store -1, 0 and 1 so we save computational resources by minimising the bit size. However reverse potentials
    #will store floating point decimals within (-0.2, 0.2) so we change the datatype here; additionally, these are fixed parameters for now, we may need to modify if we make these trainable
    def create_reverse_potentials(self, adjacency_matrix, dtype=np.float32): #noting this as a warning for potential memory in-efficiency

        reverse_potentials = np.zeros_like(adjacency_matrix, dtype=dtype) #create empty reverse potential values
        
        #creates matrix with boolean values
        mask_excitatory = (adjacency_matrix > 0) #exicted reverse potentials: 0.0 <= x <= 0.2
        mask_inhibitory = (adjacency_matrix < 0) #inhibitory reverse potentials: -0.2 <= x <= 0.0
        
        #calculating the sum tells us the amount of excitatory and inhibatory values there are
        size_excit = np.sum(mask_excitatory)
        size_inhib = np.sum(mask_inhibitory)

        #random uniform distribution of reverse potential values bounded by [-0.2, 0.2], with same output shape as adjacency matrix (either neuron_connections or sensory)
        reverse_potentials[mask_excitatory] = np.random.uniform(0.0, 0.2, size=size_excit)
        reverse_potentials[mask_inhibitory] = np.random.uniform(-0.2, 0.0, size=size_inhib)

        return reverse_potentials

    def neuron_reverse_potential_initialiser(self): #creates reverse potentials for neuron connections
        return self.create_reverse_potentials(self.neuron_connections)

    #same thing as above but using sensory connections
    def input_reverse_potential_initialiser(self): #same thing as above but creating reverse potentials for 
        return self.create_reverse_potentials(self.sensory_connections)


    def get_neuron_type(self, neuron_id): #may be overwritten by child classes i.e LTCCell
        return "motor" if neuron_id < self.output_dim else 'inter'
    
    def check_connection_values(self, src, dest, polarity, reference_size, type= "neurons"):
        #error handling for src input
        if src < 0 or src >= reference_size: #reference size can be total_neurons for add_synapse or input_dim for add_sensory_synapse
            if type == "neurons":
                print(f"Cannot add neuron connection from neuron {src} when only {reference_size} neurons exist...")
            if type == "features":
                print(f"Cannot add neuron connection from neuron {src} when only {reference_size} features exist...")
            return False 
        
        if dest < 0 or dest >= self.total_neurons:
            print(f"Cannot add connection to neuron {dest} when only {self.total_neurons} neurons exist...")
            return False 
        
        if polarity not in [-1, 1]: #must be either -1 or 1
            print(f"Cannot add connection with polarity {polarity} (expected -1 or +1)")
            return False 
        
        return True #input args are valid
    
    def add_synapse_connections(self, src, dest, polarity): #adjacency matrix for neuron connections (neuron to neuron)
        valid_input = self.check_connection_values(src, dest, polarity, self.total_neurons, type="neurons")

        if valid_input:
            self.neuron_connections[src, dest] = polarity #adds polarity connection, if two neurons have 0, they are not connected

    #remember self.sensory_connections has shape input_dim to total_neurons
    def add_sensory_synapse_connections(self, src, dest, polarity):
        valid_input = self.check_connection_values(src, dest, polarity, self.input_dim, type="features")

        if valid_input:
            self.sensory_connections[src, dest] = polarity
    
    @property
    def synapse_count(self): #returning count of internal neuron to neruon connections
        return np.sum(np.abs(self.neuron_connections))
    @property
    def sensory_synapse_count(self):
        return np.sum(np.abs(self.sensory_connections)) #return count of neurons from input to sensory
    
    def get_config(self):
        return {
            'total_neurons': self.total_neurons,
            'neuron_connections': self.neuron_connections.tolist() if self.neuron_connections is not None else None,
            'sensory_connections': self.sensory_connections.tolist() if self.sensory_connections is not None else None,
            'input_dim': self.input_dim if self.input_dim else None, 
            'output_dim': self.output_dim if self.output_dim else None
        }
    
    #TO DO, visualisation function later~


#---------------------------
#Neural Circuit Policy (NCP)
#---------------------------
class NeuralCircuitPolicy(NeuronWiring): #child class of NeuronWiring
    def __init__(self, 
                 inter_neurons, #number of inter neurons (feature extractors, sensory neurons -> inter neurons)
                 command_neurons, #number of command neurons (recurrent layer; memory/context; inter neurons -> command neurons)
                 motor_neurons, #number of motor neurons (output layer (i.e steering angle or new hidden state))
                 outgoing_sensory_neurons, #number of neurons from sensory to inter neurons
                 outgoing_inter_neurons, #number of neurons from inter to command neurons
                 num_of_recurrent_connections, #number of recurrent connections in command neuron layer
                 incoming_command_neurons, #number of incoming synapses from command to motor neurons
                 seed=24573471): #random seed for producing wiring
        
        total_neurons = inter_neurons + command_neurons + motor_neurons

        super(NeuralCircuitPolicy, self).__init__(total_neurons) #inherent NeuronWiring parent class

        self.set_output_dim(motor_neurons) #output of neuron connections is motor cells which can output either steering angle or hidden state
        self.rndm_sd = np.random.RandomState(seed) #creates random state object with a seed for individual reproducability instead of global random seeds

        #storing neuron counts
        self.num_inter_neurons = inter_neurons
        self.num_command_neurons = command_neurons
        self.num_motor_neurons = motor_neurons

        #connectivity parameters
        self.sensory_fanout = outgoing_sensory_neurons
        self.inter_fanout = outgoing_inter_neurons
        self.recurrent_command_synapses = num_of_recurrent_connections
        self.motor_fanin = incoming_command_neurons 

        #create neuron indicies, using order: [motor_neurons, command_neurons, inter_neurons] <- note that actual IDs are different i.e
        self.motor_neuron_idxs = [i for i in range(0, self.num_motor_neurons)] # [0,...m_n]
        self.command_neuron_idxs = [i for i in range(self.num_motor_neurons, (self.num_motor_neurons + self.num_command_neurons))] # [m_n,... (m_n + c_n)]
        self.inter_neuron_idxs = [i for i in range((self.num_motor_neurons + self.num_command_neurons), (self.num_motor_neurons + self.num_command_neurons + self.num_inter_neurons))] #[(m_n + c_n),... (m_n + c_n + i_n)]

        #validate connectivity parameters
        self.validate_connectivity(self.motor_fanin, self.num_command_neurons, fan_param="motor fan_in", neuron_param="command") #checking if specified output number of command neurons is larger than the actual number of command neurons
        self.validate_connectivity(self.sensory_fanout, self.num_inter_neurons, fan_param="sensory fan_out", neuron_param="inter") #check if incoming num of sensory neurons is larger than number of inter neurons
        self.validate_connectivity(self.inter_fanout, self.num_command_neurons, fan_param="inter fan_out", neuron_param="command")

    def validate_connectivity(self, fanin_fanout, num_neuron_type, fan_param, neuron_param):
        if fanin_fanout > num_neuron_type:
            raise ValueError(f"{fan_param} Paramerter is {fanin_fanout}, but there are only {num_neuron_type} {neuron_param} neurons")
    

    @property
    def layer_count(self):
        return 3 #3 layers --> inter, command, motor

    def get_neurons_of_layer(self, neuron_id): #neuron ID structure: 0 -> Inter, 1 -> Command, 2 -> Motor, which will return the corresponding indicies
        if neuron_id == 0:
            return self.inter_neuron_idxs
        if neuron_id == 1:
            return self.command_neuron_idxs
        if neuron_id == 2:
            return self.motor_neuron_idxs
        
        raise ValueError(f"Unknown id: {neuron_id}")
    
    #using index id, return the type of neuron it is
    def get_neuron_type(self, index):
        if index < self.num_motor_neurons: #because indicies start at motor neurons
            return "motor"
        
        if index < (self.num_motor_neurons + self.num_command_neurons):
            return "command"
        
        if index < (self.num_motor_neurons + self.num_command_neurons + self.num_inter_neurons):
            return "inter" 
    
        raise ValueError(f"ID {index} is higher than actual number of neuron ids: {(self.num_motor_neurons + self.num_command_neurons + self.num_inter_neurons) - 1}") #else raise value error that index is too high

#---------------------------Building connections---------------------------
    #input -> sensory -> inter ->....
    def build_sensory_to_inter_connections(self):
        unreachable_inter_neurons = [neuron_id for neuron_id in self.inter_neuron_idxs] #initialise with all inter neuron ids; ids that are used will be removed from the list dynamically/randomly

        for src in self.sensory_neuron_idxs: #which is initialised in 'build' method below
            #selecting random indexes from inter neurons as 'dst' in adjacency matrix, replace=False means it can't randomly select the same index twice, so values are unique
            for dst in self.rndm_sd.choice(self.inter_neuron_idxs, size=self.sensory_fanout, replace=False): 
                if dst in unreachable_inter_neurons:
                    unreachable_inter_neurons.remove(dst) #remove from being unreachable
                
                polarity = self.rndm_sd.choice([-1, 1]) #random excitatory/inhabitory connection
                self.add_sensory_synapse_connections(src, dst, polarity) #add to adjacency matrix
        
        #if some inter neurons are not connected, connect them:
        mean_inter_neuron_fanin = int(self.num_sensory_neurons * self.sensory_fanout / self.num_inter_neurons) 
        mean_inter_neuron_fanin = np.clip(mean_inter_neuron_fanin, 1, self.num_sensory_neurons) 

        for dest in unreachable_inter_neurons: #looping through remaining neurons
            for src in self.rndm_sd.choice(self.sensory_neuron_idxs, size=mean_inter_neuron_fanin, replace=False):
                polarity = self.rndm_sd.choice([-1, 1])
                self.add_sensory_synapse_connections(src, dest, polarity)

    #sensory -> inter -> commands ->....
    def build_inter_to_command_connections(self):
        unreachable_command_neurons = [neuron_id for neuron_id in self.command_neuron_idxs]

        for src in self.inter_neuron_idxs:
            for dst in self.rndm_sd.choice(self.command_neuron_idxs, size=self.inter_fanout, replace=False):
                if dst in unreachable_command_neurons:
                    unreachable_command_neurons.remove(dst)
                polarity = self.rndm_sd.choice([-1, 1])
                self.add_synapse_connections(src, dst, polarity) #add to regular neurons connection adjacency matrix
        
        mean_command_neuron_fain = int(self.num_inter_neurons * self.inter_fanout / self.num_command_neurons)
        mean_command_neuron_fain = np.clip(mean_command_neuron_fain, 1, self.num_command_neurons)

        for dst in unreachable_command_neurons:
            for src in self.rndm_sd.choice(self.inter_neuron_idxs, size=mean_command_neuron_fain, replace=False):
                polarity = self.rndm_sd.choice([-1, 1])
                self.add_synapse_connections(src, dst, polarity)

    #randomly map connectivity between command neurons for recurrent processing
    def build_recurrent_command_layer(self):
        for i in range(self.recurrent_command_synapses):
            src = self.rndm_sd.choice(self.command_neuron_idxs)
            dst = self.rndm_sd.choice(self.command_neuron_idxs)
            polarity = self.rndm_sd.choice([-1, 1])
            self.add_synapse_connections(src, dst, polarity)

    #inter -> commands -> motor ->....
    def build_command_to_motor_layer(self):
        unreachable_command_neurons = [neuron_id for neuron_id in self.command_neuron_idxs] #using command neuron idxs so that random command neurons connect to motor neurons

        for dst in self.motor_neuron_idxs:
            for src in self.rndm_sd.choice(self.command_neuron_idxs, size=self.motor_fanin, replace=False):
                if src in unreachable_command_neurons:
                    unreachable_command_neurons.remove(src)
                polarity = self.rndm_sd.choice([-1, 1])
                self.add_synapse_connections(src, dst, polarity)
        
        mean_command_fanout = int(self.num_motor_neurons * self.motor_fanin / self.num_command_neurons)
        mean_command_fanout = np.clip(mean_command_fanout, 1, self.num_motor_neurons)

        for src in unreachable_command_neurons:
            for dst in self.rndm_sd.choice(self.motor_neuron_idxs, size=mean_command_fanout, replace=False):
                polarity = self.rndm_sd.choice([-1, 1])
                self.add_synapse_connections(src, dst, polarity)

    #build sensory neurons here
    def build(self, input_shape):

        #initialise wiring
        super().build(input_shape) #input_dim 
        self.num_sensory_neurons = self.input_dim #which comes from NeuronWiring
        self.sensory_neuron_idxs = [i for i in range(0, self.num_sensory_neurons)] #even tho motor indxs start from 0, this is not included in the layer and acts more like a gate and sensory neurons has its own sensory_connections (adjacency matrix)

        #after sensory neurons, build the connections; sensory -> inter -> command -> motor
        self.build_sensory_to_inter_connections()
        self.build_inter_to_command_connections()
        self.build_recurrent_command_layer()
        self.build_command_to_motor_layer()

    def get_config(self):
        return {
            'inter_neurons_ids': self.inter_neuron_idxs,
            'command_neuron_ids': self.command_neuron_idxs,
            'motor_neuron_ids': self.motor_neuron_idxs,
            'sensory_fanout': self.sensory_fanout,
            'inter_fanout': self.inter_fanout,
            'num_recurrent_connections': self.recurrent_command_synapses,
            'motor_fanin': self.motor_fanin,
            'seed': self.rndm_sd.seed()
        }