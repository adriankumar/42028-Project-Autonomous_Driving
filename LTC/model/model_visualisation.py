import networkx as nx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

#function to plot the adjacency matrices 
def plot_adjacency_matrices(wiring, figsize=(12, 5), save_path=None):
    if not wiring.is_built():
        print("wiring is not initialised, build it first with input shape")
        return
    
    plt.figure(figsize=figsize)
    
    #plot neuron connections matrix
    plt.subplot(1, 2, 1)
    plt.imshow(wiring.get_NAM, cmap='PiYG', vmin=-1, vmax=1) #pink to green, pink is inhibitory, green is excitatory
    plt.colorbar(label='+1 Excitatory, -1 Inhibitory, 0 No connection')
    plt.title('Internal Neuron Connections')
    plt.xlabel('destination neuron')
    plt.ylabel('source neuron')
    plt.tight_layout()
    
    #create correct neuron labels by type, recall index order is motor -> command -> inter
    neuron_labels = []
    for i in range(wiring.internal_neuron_total):
        neuron_type = wiring.get_neuron_type(i)
        if neuron_type == "motor":
            idx = wiring.motor_neurons.index(i)
            neuron_labels.append(f"m{idx}")
        elif neuron_type == "command":
            idx = wiring.command_neurons.index(i)
            neuron_labels.append(f"c{idx}")
        elif neuron_type == "inter":
            idx = wiring.interneurons.index(i)
            neuron_labels.append(f"i{idx}")
    
    plt.xticks(range(wiring.internal_neuron_total), neuron_labels, rotation=45)
    plt.yticks(range(wiring.internal_neuron_total), neuron_labels)
    plt.xlabel('destination neuron')
    plt.ylabel('source neuron')
    
    #plot sensory connections matrix
    plt.subplot(1, 2, 2)
    plt.imshow(wiring.get_SAM, cmap='PiYG', vmin=-1, vmax=1)
    plt.colorbar(label='+1 Excitatory, -1 Inhibitory, 0 No connection')
    plt.title('Sensory to neruon connections')
    
    #create sensory labels
    sensory_labels = [f"s{i}" for i in range(wiring.input_dim)]
    
    plt.yticks(range(wiring.input_dim), sensory_labels)
    plt.xticks(range(wiring.internal_neuron_total), neuron_labels, rotation=45)
    plt.xlabel('destination neuron')
    plt.ylabel('source sensory')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

#helper wiring graph function taken from official github, refactored into pytorch: https://github.com/mlech26l/ncps/blob/master/ncps/wirings/wirings.py
def create_wiring_graph(wiring):
    DG = nx.DiGraph()
    
    #add neurons as nodes with exact indices
    for i in range(wiring.internal_neuron_total):
        neuron_type = wiring.get_neuron_type(i)
        if neuron_type == "motor":
            idx = wiring.motor_neurons.index(i)
            label = f"m{idx}"
        elif neuron_type == "command":
            idx = wiring.command_neurons.index(i)
            label = f"c{idx}"
        else:  # inter
            idx = wiring.interneurons.index(i)
            label = f"i{idx}"
        DG.add_node(i, neuron_type=neuron_type, label=label)
    
    #add sensory neurons as nodes
    for i in range(wiring.input_dim):
        DG.add_node(f"s{i}", neuron_type="sensory", label=f"s{i}")
    
    #add sensory connections exactly as in adjacency matrix
    for src in range(wiring.input_dim):
        for dest in range(wiring.internal_neuron_total):
            if wiring.sensory_adjacency_matrix[src, dest] != 0:
                polarity = "excitatory" if wiring.sensory_adjacency_matrix[src, dest] > 0 else "inhibitory"
                DG.add_edge(
                    f"s{src}",
                    dest,
                    polarity=polarity,
                    weight=wiring.sensory_adjacency_matrix[src, dest]
                )
    
    #add neuron connections exactly as in adjacency matrix
    for src in range(wiring.internal_neuron_total):
        for dest in range(wiring.internal_neuron_total):
            if wiring.neuron_adjacency_matrix[src, dest] != 0:
                polarity = "excitatory" if wiring.neuron_adjacency_matrix[src, dest] > 0 else "inhibitory"
                DG.add_edge(
                    src,
                    dest,
                    polarity=polarity,
                    weight=wiring.neuron_adjacency_matrix[src, dest]
                )
    
    return DG

#plot neural wiring using create_wiring_graph helper function
def view_neural_wiring(wiring, figsize=(15, 12), save_path=None, show=False):
    if not wiring.is_built():
        print("wiring is not initialised, build it first with input shape")
        return
    
    #stup colours
    neuron_colours = {
        "inter": "tab:blue",
        "command": "tab:orange",
        "motor": "tab:purple",
        "sensory": "tab:olive",
    }
    
    synapse_colours = {
        "excitatory": "tab:green", 
        "inhibitory": "tab:red"
    }
    
    #create the graph
    DG = create_wiring_graph(wiring)
    
    plt.figure(figsize=figsize)
    
    #spiral layout for better visualisation
    pos = nx.spiral_layout(DG, scale=2.0)
    
    #draw neurons
    for neuron_type, color in neuron_colours.items():
        if neuron_type == "sensory":
            nodes = [n for n in DG.nodes() if isinstance(n, str) and n.startswith("s")] #sensory neurons
        else:
            nodes = [n for n, d in DG.nodes(data=True) if d.get('neuron_type') == neuron_type] #inter, command and motor neurons
        nx.draw_networkx_nodes(DG, pos, nodes, node_color=color, node_size=500)
    
    #draw edges/synapses with correct colours for polarity
    for edge_type, color in synapse_colours.items():
        edges = [(u, v) for u, v, d in DG.edges(data=True) if d.get('polarity') == edge_type]
        nx.draw_networkx_edges(DG, pos, edges, edge_color=color, arrows=True, 
                              connectionstyle='arc3,rad=0.1', width=1.5)
    
    labels = {}
    for node in DG.nodes():
        if isinstance(node, str) and node.startswith("s"):
            labels[node] = node
        else:
            labels[node] = DG.nodes[node]['label']
    
    nx.draw_networkx_labels(DG, pos, labels=labels)
    
    legend_patches = []
    for neuron_type, color in neuron_colours.items():
        label = f"{neuron_type.capitalize()} neurons"
        legend_patches.append(mpatches.Patch(color=color, label=label))
    
    for conn_type, color in synapse_colours.items():
        legend_patches.append(mpatches.Patch(color=color, label=f"{conn_type} connection"))
    
    plt.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.title("Neural Circuit Policy Architecture")
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close()

#helper function to print matrix connections in terminal for manual validation
def print_matrix_connections(wiring):
    print("===== SENSORY CONNECTIONS =====")
    for src in range(wiring.input_dim):
        for dest in range(wiring.internal_neuron_total):
            if wiring.sensory_adjacency_matrix[src, dest] != 0:
                polarity = "+" if wiring.sensory_adjacency_matrix[src, dest] > 0 else "-"
                neuron_type = wiring.get_neuron_type(dest)
                print(f"sensory_{src} --{polarity}--> {neuron_type}_{dest}")
    
    print("\n===== NEURON CONNECTIONS =====")
    for src in range(wiring.internal_neuron_total):
        src_type = wiring.get_neuron_type(src)
        for dest in range(wiring.internal_neuron_total):
            if wiring.neuron_adjacency_matrix[src, dest] != 0:
                polarity = "+" if wiring.neuron_adjacency_matrix[src, dest] > 0 else "-"
                dest_type = wiring.get_neuron_type(dest)
                print(f"{src_type}_{src} --{polarity}--> {dest_type}_{dest}")