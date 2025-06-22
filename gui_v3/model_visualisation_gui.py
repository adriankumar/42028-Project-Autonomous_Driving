import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg

#visual configuration for gui integration
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
CURVE_DEPTH = 100
RECURRENT_ARC_STRENGTH = 30

class ModelVisualisation:
    def __init__(self, model):
        #use already loaded model from model_handler
        self.model = model
        self.wiring = self.model.ltc_cell.wiring
        
        #extract actual neuron counts from loaded model
        self.num_inter_neurons = len(self.wiring.interneurons)
        self.num_command_neurons = len(self.wiring.command_neurons)
        self.num_motor_neurons = len(self.wiring.motor_neurons)
        
        #get actual adjacency matrix for real connections
        self.adjacency_matrix = self.wiring.neuron_adjacency_matrix
        
        self.neurons = []
        self.connections = []
        self.neuron_map = {}
        
        #current model state for live updating
        self.current_hidden_states = np.zeros(self.model.ltc_cell.internal_neuron_size)
        self.current_synaptic_weights = None

        #precompute colour gradient for efficient neuron activation mapping
        self.true_colours_hex = [
            '#FF9980', '#FF8E8A', '#FF8394', '#FF789E', '#FF6DA8', '#FF62B2', '#FF57BC', '#FF4CC6',
            '#F941D0', '#F236DA', '#EA2BE4', '#E220EE', '#D916F8', '#CF0DFF', '#C40DFF', '#BA0DFF',
            '#B00DFF', '#A60DFF', '#9C0DFF', '#920DFF', '#880DFF', '#7E0DFF', '#740DFF', '#6A0DFF'
        ]
        
        #convert hex to rgb tuples for matplotlib efficiency
        self.colour_palette = []
        for hex_colour in self.true_colours_hex:
            #convert hex to rgb (0-1 range for matplotlib)
            hex_colour = hex_colour.lstrip('#')
            rgb = tuple(int(hex_colour[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            self.colour_palette.append(rgb)

        #setup visualisation structure once
        self.generate_neurons()
        self.generate_connections()
        
        #create matplotlib figure and visual elements once
        self._create_figure()
        
    def _create_figure(self):
        #create figure and axis once during initialisation
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(CANVAS_WIDTH/100, CANVAS_HEIGHT/100), dpi=100)
        self.fig.patch.set_facecolor('#1a1a1c')
        self.ax.set_facecolor('#000000')
        
        #store references to visual elements for efficient updating
        self.connection_lines = {}
        self.neuron_circles = {}
        
        #draw connections first (so they appear behind neurons)
        for conn in self.connections:
            source_neuron = self.neuron_map[conn['source']]
            target_neuron = self.neuron_map[conn['target']]
            x1, y1 = source_neuron['x'], source_neuron['y']
            x2, y2 = target_neuron['x'], target_neuron['y']
            
            #get colour and opacity based on connection weight (static)
            colour, opacity = self._get_connection_colour_and_opacity(conn['weight'])
            
            #determine if connection is recurrent
            is_recurrent = source_neuron['type'] == 'command' and target_neuron['type'] == 'command'
            
            if is_recurrent:
                #create outward arc for recurrent connections
                arc_mid_x = (x1 + x2) / 2 + RECURRENT_ARC_STRENGTH
                control_x1 = arc_mid_x
                control_y1 = y1
                control_x2 = arc_mid_x
                control_y2 = y2
            else:
                #standard bezier curve for layer-to-layer connections
                control_x1 = x1 + (x2 - x1) * 0.35
                control_y1 = y1
                control_x2 = x2 - (x2 - x1) * 0.35
                control_y2 = y2
            
            #create bezier curve using multiple segments for all connections
            t_values = np.linspace(0, 1, 20)
            x_points = []
            y_points = []
            
            for t in t_values:
                #cubic bezier curve
                x = (1-t)**3 * x1 + 3*(1-t)**2*t * control_x1 + 3*(1-t)*t**2 * control_x2 + t**3 * x2
                y = (1-t)**3 * y1 + 3*(1-t)**2*t * control_y1 + 3*(1-t)*t**2 * control_y2 + t**3 * y2
                x_points.append(x)
                y_points.append(y)
            
            line, = self.ax.plot(x_points, y_points, color=colour, alpha=opacity, linewidth=1.0, zorder=1)
            
            #store reference to the line object
            self.connection_lines[(conn['src_model_idx'], conn['dst_model_idx'])] = line
        
        #draw neurons on top of connections
        for neuron in self.neurons:
            x, y = neuron['x'], neuron['y']
            
            #initial neutral colour
            fill_colour = (0.5, 0.5, 0.5)  #default grey
            
            #neuron type specific radius
            if neuron['type'] == 'inter':
                radius = 6
            elif neuron['type'] == 'command':
                radius = 6
            else:  #motor
                radius = 8
            
            #draw neuron circle and store reference - no edge colour
            circle = patches.Circle((x, y), radius, facecolor=fill_colour, linewidth=0, zorder=2)
            self.ax.add_patch(circle)
            self.neuron_circles[neuron['model_idx']] = circle
        
        #add colour gradient key for normalised neural state
        self._create_colour_key()
        
        #add title and labels
        self.ax.text(CANVAS_WIDTH/2, CANVAS_HEIGHT - 20, 'LTC Neural Activity', 
               ha='center', va='center', fontsize=10, color='white', weight='bold')
        
        self.ax.text(CANVAS_WIDTH * 0.25, 15, 'Inter', ha='center', va='center', 
               fontsize=8, color='#aaa')
        self.ax.text(CANVAS_WIDTH * 0.55, 15, 'Command', ha='center', va='center', 
               fontsize=8, color='#aaa')
        self.ax.text(CANVAS_WIDTH * 0.85, 15, 'Motor', ha='center', va='center', 
               fontsize=8, color='#aaa')
        
        #set axis properties
        self.ax.set_xlim(0, CANVAS_WIDTH)
        self.ax.set_ylim(0, CANVAS_HEIGHT)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        #remove margins
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    
    def _create_colour_key(self):
        #create gradient colour key showing normalised neural state mapping
        key_width = 150
        key_height = 15
        key_x = 600  #position from left edge - moved right
        key_y = 530  #position from bottom - moved up to upper left area
        
        #create gradient segments
        num_segments = 100
        segment_width = key_width / num_segments
        
        for i in range(num_segments):
            #map segment to activation value (-1 to +1)
            activation_value = (i / (num_segments - 1)) * 2 - 1  #maps 0-99 to -1 to +1
            
            #get colour for this activation value using same function as neurons
            colour = self._get_colour_for_activation(activation_value)
            
            #create rectangle for this segment
            rect_x = key_x + i * segment_width
            rect = patches.Rectangle((rect_x, key_y), segment_width, key_height, 
                                facecolor=colour, linewidth=0, zorder=2)
            self.ax.add_patch(rect)
        
        #add labels
        self.ax.text(key_x, key_y - 10, '-1', ha='center', va='top', 
            fontsize=8, color='white')
        self.ax.text(key_x + key_width/2, key_y - 10, '0', ha='center', va='top', 
            fontsize=8, color='white')
        self.ax.text(key_x + key_width, key_y - 10, '+1', ha='center', va='top', 
            fontsize=8, color='white')
        
        #add title for colour key
        self.ax.text(key_x + key_width/2, key_y + key_height + 15, 'Normalised neural state', 
            ha='center', va='bottom', fontsize=9, color='white', weight='bold')

    def _get_colour_for_activation(self, activation):
        #helper function to get colour for specific activation value (used for gradient key)
        activation = float(activation)
        normalised_activation = np.tanh(activation)
        
        #map activation to continuous colour index
        continuous_index = (1.0 - normalised_activation) * 0.5 * (len(self.colour_palette) - 1)
        continuous_index = np.clip(continuous_index, 0, len(self.colour_palette) - 1)
        
        #get adjacent colour indices
        lower_idx = int(np.floor(continuous_index))
        upper_idx = min(lower_idx + 1, len(self.colour_palette) - 1)
        
        #interpolation factor
        t = continuous_index - lower_idx
        
        #interpolate between adjacent colours
        lower_colour = self.colour_palette[lower_idx]
        upper_colour = self.colour_palette[upper_idx]
        
        interpolated_colour = tuple(
            lower_colour[i] * (1 - t) + upper_colour[i] * t
            for i in range(3)
        )
        
        return interpolated_colour

    def generate_neurons(self):
        #generate positions for all neurons using actual model configuration
        #inter neurons (leftmost layer)
        x_base_inter = CANVAS_WIDTH * 0.25
        y_step_inter = CANVAS_HEIGHT / (self.num_inter_neurons + 1)
        for i, neuron_idx in enumerate(self.wiring.interneurons):
            y = y_step_inter * (i + 1)
            #apply parabolic curve to x-coordinate
            normalised_y = (y - CANVAS_HEIGHT / 2) / (CANVAS_HEIGHT / 2)
            x_offset = CURVE_DEPTH * (normalised_y ** 2)
            x = x_base_inter - x_offset

            neuron = {"id": f"inter_{i}", "model_idx": neuron_idx, "type": "inter", "x": x, "y": y}
            self.neurons.append(neuron)
            self.neuron_map[neuron['id']] = neuron

        #command neurons (middle layer)
        x_base_command = CANVAS_WIDTH * 0.55
        y_step_command = CANVAS_HEIGHT / (self.num_command_neurons + 1)
        for i, neuron_idx in enumerate(self.wiring.command_neurons):
            y = y_step_command * (i + 1)
            normalised_y = (y - CANVAS_HEIGHT / 2) / (CANVAS_HEIGHT / 2)
            x_offset = CURVE_DEPTH * (normalised_y ** 2)
            x = x_base_command - x_offset
            
            neuron = {"id": f"command_{i}", "model_idx": neuron_idx, "type": "command", "x": x, "y": y}
            self.neurons.append(neuron)
            self.neuron_map[neuron['id']] = neuron

        #motor neurons (rightmost layer)
        x_base_motor = CANVAS_WIDTH * 0.85
        y_step_motor = CANVAS_HEIGHT / (self.num_motor_neurons + 1)
        for i, neuron_idx in enumerate(self.wiring.motor_neurons):
            neuron = {
                "id": f"motor_{i}", "model_idx": neuron_idx, "type": "motor",
                "x": x_base_motor, "y": y_step_motor * (i + 1)
            }
            self.neurons.append(neuron)
            self.neuron_map[neuron['id']] = neuron

    def generate_connections(self):
        #generate connections based on actual adjacency matrix
        for src_idx in range(len(self.adjacency_matrix)):
            for dst_idx in range(len(self.adjacency_matrix)):
                weight = self.adjacency_matrix[src_idx, dst_idx]
                if weight != 0:  #only create connection if it exists in model
                    #find corresponding visual neurons
                    src_neuron = self._find_neuron_by_model_idx(src_idx)
                    dst_neuron = self._find_neuron_by_model_idx(dst_idx)
                    
                    if src_neuron and dst_neuron:
                        connection = {
                            "source": src_neuron['id'], 
                            "target": dst_neuron['id'],
                            "weight": float(weight),
                            "src_model_idx": src_idx,
                            "dst_model_idx": dst_idx
                        }
                        self.connections.append(connection)

    def _find_neuron_by_model_idx(self, model_idx):
        #helper function to find visual neuron by model index
        for neuron in self.neurons:
            if neuron['model_idx'] == model_idx:
                return neuron
        return None

    #alternative, use exact gradient colours as is
    # def _get_neuron_colour(self, activation):
    #     #convert neuron activation to rgb colour using custom gradient
    #     #ensure activation is a regular python float
    #     activation = float(activation)
        
    #     #normalise using tanh to -1 to 1 range
    #     normalised_activation = np.tanh(activation)
        
    #     #map activation to colour index (orange=+1, purple=-1, pink=0)
    #     #reverse mapping: +1 -> index 0 (orange), -1 -> index 23 (purple)
    #     colour_index = (1.0 - normalised_activation) * 0.5 * (len(self.colour_palette) - 1)
        
    #     #clamp to valid range and convert to integer
    #     colour_index = int(np.clip(colour_index, 0, len(self.colour_palette) - 1))
        
    #     return self.colour_palette[colour_index]

    #convert neuron activation to rgb colour using custom gradient with interpolation
    def _get_neuron_colour(self, activation):
        activation = float(activation)
        normalised_activation = np.tanh(activation)
        
        #map activation to continuous colour index
        continuous_index = (1.0 - normalised_activation) * 0.5 * (len(self.colour_palette) - 1)
        continuous_index = np.clip(continuous_index, 0, len(self.colour_palette) - 1)
        
        #get adjacent colour indices
        lower_idx = int(np.floor(continuous_index))
        upper_idx = min(lower_idx + 1, len(self.colour_palette) - 1)
        
        #interpolation factor
        t = continuous_index - lower_idx
        
        #interpolate between adjacent colours
        lower_colour = self.colour_palette[lower_idx]
        upper_colour = self.colour_palette[upper_idx]
        
        interpolated_colour = tuple(
            lower_colour[i] * (1 - t) + upper_colour[i] * t
            for i in range(3)
        )
        
        return interpolated_colour

    def _get_connection_colour_and_opacity(self, weight):
        #determine connection colour and opacity based on weight
        if weight > 0:
            colour = "#00d13f"  #green for excitatory
        else:
            colour = "#d10072"  #red for inhibitory
            
        #set opacity based on weight magnitude for sparsity visualisation
        opacity = min(abs(weight), 1.0)
        
        return colour, opacity

    def update_neuron_states(self, hidden_states, synaptic_weights=None):
        #update current hidden states for live visualisation
        #handle batch dimension properly - model returns [1, 38] but we need [38]
        if isinstance(hidden_states, torch.Tensor):
            if hidden_states.dim() > 1:
                #remove batch dimension if present [1, 38] -> [38]
                hidden_states = hidden_states.squeeze(0)
            self.current_hidden_states = hidden_states.cpu().detach().numpy()
        else:
            #handle numpy array with potential batch dimension
            hidden_states = np.array(hidden_states)
            if hidden_states.ndim > 1:
                hidden_states = hidden_states.squeeze(0)
            self.current_hidden_states = hidden_states
        
        #store synaptic weights if provided
        if synaptic_weights is not None:
            self.current_synaptic_weights = synaptic_weights
        
        #update visual elements immediately without recreating anything
        self._update_visual_elements()
        
    def _update_visual_elements(self):
        #update only the colours of existing visual elements - no recreation
        #update neuron colours based on current hidden states
        for neuron_idx, circle in self.neuron_circles.items():
            if neuron_idx < len(self.current_hidden_states):
                activation = self.current_hidden_states[neuron_idx]
                fill_colour = self._get_neuron_colour(activation)
                circle.set_facecolor(fill_colour)
        
        #update connection transparency based on synaptic weights
        if self.current_synaptic_weights is not None:
            #scaling factor to enhance visibility of small synaptic values
            transparency_scale = 10.0  #adjust this value as needed
            
            for conn in self.connections:
                src_idx = conn['src_model_idx']
                dst_idx = conn['dst_model_idx']
                
                #check if indices are valid for synaptic weights matrix
                if (src_idx < self.current_synaptic_weights.shape[0] and 
                    dst_idx < self.current_synaptic_weights.shape[1]):
                    
                    synaptic_value = self.current_synaptic_weights[src_idx, dst_idx]
                    #scale and clip for better visibility
                    alpha = min(abs(synaptic_value) * transparency_scale, 1.0)
                    
                    if (src_idx, dst_idx) in self.connection_lines:
                        self.connection_lines[(src_idx, dst_idx)].set_alpha(alpha)

    def get_visualisation_image(self):
        #convert current figure to PIL image without recreating anything
        #use existing figure and just render it to image
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()
        
        #get image data efficiently
        buf = canvas.buffer_rgba()
        img_array = np.asarray(buf)
        img = Image.fromarray(img_array)
        
        return img

    def cleanup(self):
        #cleanup matplotlib figure properly
        if hasattr(self, 'fig'):
            plt.close(self.fig)