import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pk
import seaborn as sns
import copy


def label_conversations_with_frag_score(path):
    conversation_network_to_graph_id = pk.load(open(f"{path}conversation_network_to_graph_id.pk", "rb"))
    tweet_to_network_id = {}
    for network in conversation_network_to_graph_id.keys():
        for node in network.nodes:
            tweet_to_network_id[node] = conversation_network_to_graph_id[network]
    relevant_modified_hyper_network_with_attributes_frag_score = nx.DiGraph()
    for graph in conversation_network_to_graph_id.keys():
        relevant_modified_hyper_network_with_attributes_frag_score.add_nodes_from(graph.nodes)
        relevant_modified_hyper_network_with_attributes_frag_score.add_edges_from(graph.edges)

    return relevant_modified_hyper_network_with_attributes_frag_score


def make_dyadic_inference(hyper_network, aggregated_labels_dict):
    relevant_modified_hyper_network_with_attributes_frag_score = copy.deepcopy(hyper_network)
    aggregated_labels = copy.deepcopy(aggregated_labels_dict)

    nodes_dict = {}
    nodes_neighbour_label = {}
    distinct_labels = set()
    conditional_probs_dyad_hyper_network = {}
    for node in relevant_modified_hyper_network_with_attributes_frag_score.nodes:
        if relevant_modified_hyper_network_with_attributes_frag_score.in_degree(node) != 0:
            if aggregated_labels[node] in ["L3", "L4"]:                
                node_in_degree_temp = 0
                for edge in relevant_modified_hyper_network_with_attributes_frag_score.in_edges(node):
                    if aggregated_labels[edge[0]] in ["L3", "L4"]:   
                        node_in_degree_temp += 1
                if node_in_degree_temp > 0:
                    distinct_labels.add(aggregated_labels[node])
                    node_in_degree = node_in_degree_temp
                    nodes_dict[aggregated_labels[node]] = nodes_dict.get(aggregated_labels[node], 0) + node_in_degree

                    for edge in relevant_modified_hyper_network_with_attributes_frag_score.in_edges(node):
                        if aggregated_labels[edge[0]] in ["L3", "L4"]:   
                            nodes_neighbour_label[aggregated_labels[node]] = nodes_neighbour_label.get(aggregated_labels[node], []) + [aggregated_labels[edge[0]]]
                
    givens = list(distinct_labels)
    probs = list(distinct_labels)
    for condition1 in givens:
        for condition2 in probs:
            conditional_probs_dyad_hyper_network[condition2+"|"+condition1] = sum([1 for i in nodes_neighbour_label[condition1] if i == condition2]) / nodes_dict[condition1]


    df_conditional_probs_dyad_hyper_network_heatmap = pd.DataFrame(columns=["L3", "L4"], index=["L3", "L4"], dtype="float")

    for likelihood in conditional_probs_dyad_hyper_network:
        indeces = likelihood.split("|")
        df_conditional_probs_dyad_hyper_network_heatmap.loc[indeces[0]][indeces[1]] = conditional_probs_dyad_hyper_network[likelihood]

    return df_conditional_probs_dyad_hyper_network_heatmap


def plot_dyadic_interaction(conditional_prob_df, subject, path):
    df_conditional_probs_dyad_hyper_network_heatmap = copy.deepcopy(conditional_prob_df)
    topic = copy.deepcopy(subject)

    fig, ax = plt.subplots(figsize=(8,6)) 
    ax = sns.heatmap(df_conditional_probs_dyad_hyper_network_heatmap, annot=True, cmap="Greens", vmin=0, vmax=1)
    ax.invert_yaxis()
    plt.xlabel("Given a node with this viewpoint")
    plt.ylabel("The likelihood of a reply with this viewpoint")
    #Subject is either Immigration or DST
    ax.set_title(topic)

    plt.savefig(f"{path}dyadic_interactions.pdf", bbox_inches='tight')


if __name__ = "__main__":
    parser = argparse.ArgumentParser(description="Enter args")
    parser.add_argument('--conversation_network_to_graph_id_path', help='Enter the path to load conversation_network -> graph_id dictionary', type=str, required=True)
    parser.add_argument('--aggregated_labels_path', help='Enter the path to aggregated labels', type=str, required=True)
    parser.add_argument('--subject', help='Enter the topic name [DST/Immigration]', type=str, required=True)
    parser.add_argument('--dyadic_interaction_plot_path', help='Enter the path to save the dyadic interaction plot', type=str, required=True)

    args = parser.parse_args()  

    conversation_network_to_graph_id_path = args.conversation_network_to_graph_id_path
    aggregated_labels = pk.load(open(args.aggregated_labels_path, "rb"))
    subject = args.subject
    dyadic_interaction_plot_path = args.dyadic_interaction_plot_path

    relevant_modified_hyper_network_with_attributes_frag_score = label_conversations_with_frag_score(conversation_network_to_graph_id_path)
    df_conditional_probs_dyad_hyper_network_heatmap = make_dyadic_inference(relevant_modified_hyper_network_with_attributes_frag_score, aggregated_labels)
    plot_dyadic_interaction(df_conditional_probs_dyad_hyper_network_heatmap, subject, dyadic_interaction_plot_path)