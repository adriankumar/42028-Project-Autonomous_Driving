import networkx as nx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from wiring_class import NeuralCircuitPolicy

#function to create the networkx graph from wiring
def create_wiring_graph(wiring):
    DG = nx.DiGraph()
    
    #add neurons as nodes
    for i in range(wiring.total_neurons):
        neuron_type = wiring.get_neuron_type(i)
        DG.add_node("neuron_{:d}".format(i), neuron_type=neuron_type)
    
    #add sensory neurons as nodes
    for i in range(wiring.input_dim):
        DG.add_node("sensory_{:d}".format(i), neuron_type="sensory")
    
    #add sensory connections as edges
    for src in range(wiring.input_dim):
        for dest in range(wiring.total_neurons):
            if wiring.sensory_connections[src, dest] != 0:
                polarity = "excitatory" if wiring.sensory_connections[src, dest] > 0 else "inhibitory"
                DG.add_edge(
                    "sensory_{:d}".format(src),
                    "neuron_{:d}".format(dest),
                    polarity=polarity,
                )
    
    #add neuron connections as edges
    for src in range(wiring.total_neurons):
        for dest in range(wiring.total_neurons):
            if wiring.neuron_connections[src, dest] != 0:
                polarity = "excitatory" if wiring.neuron_connections[src, dest] > 0 else "inhibitory"
                DG.add_edge(
                    "neuron_{:d}".format(src),
                    "neuron_{:d}".format(dest),
                    polarity=polarity,
                )
    
    return DG

#function to plot the graph
def plot_wiring(wiring, figsize=(10, 8), layout="shell", save_path=None, show=False):
    if not wiring.is_initialised():
        print("wiring is not initialised, build it first with input shape")
        return
    
    #setup colours for neuron types and synapses
    neuron_colors = {
        "inter": "tab:blue",
        "command": "tab:orange",
        "motor": "tab:purple",
        "sensory": "tab:olive",
    }
    
    synapse_colors = {
        "excitatory": "tab:green", 
        "inhibitory": "tab:red"
    }
    
    #create graph
    DG = create_wiring_graph(wiring)
    
    #setup figure
    plt.figure(figsize=figsize)
    
    #get layout function
    layouts = {
        "shell": nx.shell_layout,
        "kamada": nx.kamada_kawai_layout,
        "circular": nx.circular_layout,
        "random": nx.random_layout,
        "spring": nx.spring_layout,
        "spectral": nx.spectral_layout,
        "spiral": nx.spiral_layout,
    }
    
    layout_func = layouts.get(layout, nx.shell_layout)
    pos = layout_func(DG)
    
    #create legend patches
    legend_patches = []
    for neuron_type, color in neuron_colors.items():
        label = "{}{} neurons".format(neuron_type[0].upper(), neuron_type[1:])
        legend_patches.append(mpatches.Patch(color=color, label=label))
    
    #draw neurons
    for i in range(wiring.total_neurons):
        node_name = "neuron_{:d}".format(i)
        neuron_type = DG.nodes[node_name]["neuron_type"]
        neuron_color = neuron_colors.get(neuron_type, "tab:blue")
        nx.draw_networkx_nodes(DG, pos, [node_name], node_color=neuron_color)
    
    #draw sensory neurons
    for i in range(wiring.input_dim):
        node_name = "sensory_{:d}".format(i)
        nx.draw_networkx_nodes(DG, pos, [node_name], node_color=neuron_colors["sensory"])
    
    #draw edges
    for node1, node2, data in DG.edges(data=True):
        polarity = data["polarity"]
        edge_color = synapse_colors[polarity]
        nx.draw_networkx_edges(DG, pos, [(node1, node2)], edge_color=edge_color)
    
    #add labels
    nx.draw_networkx_labels(DG, pos)
    
    #add legend and title
    plt.legend(handles=legend_patches, loc="upper right")
    plt.title("neural circuit policy architecture")
    plt.tight_layout()
    
    #save if path provided
    if save_path:
        plt.savefig(save_path)
    
    #show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return