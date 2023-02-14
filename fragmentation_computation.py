import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import seaborn as sns
import argparse
import copy
import pickle as pk


def create_viewpoint_graphs(networks_array, tweet_author_dict, aggregated_labels_dict, conversation_network_to_graph_id__path_to_save):
    relevant_network_array = copy.deepcopy(networks_array)
    tweet_author = copy.deepcopy(tweet_author_dict)
    aggregated_labels = copy.deepcopy(aggregated_labels_dict)

    viewpoint_networks = []
    multi_graph_key_holder = {}
    viewpoint_network_to_graph_id = {}
    conversation_network_to_graph_id = {}

    index = 0

    for graph_id, G in enumerate(relevant_network_array):
        U = nx.MultiDiGraph()
        N = nx.Graph()
        multi_graph_key_holder[graph_id] = {}
        for n,d in dict(G.out_degree).items():
            if d == 0:
                root = n
                break
        
        N.add_node(tweet_author[root])
        
        iterations = [root]
        for node in iterations:
            N.add_node(tweet_author[node])
            source = tweet_author[node]
            neighbours = [i for i in nx.Graph.neighbors(G.reverse(),node)]
            iterations += neighbours

            seen = [tweet_author[i] for i in nx.Graph.neighbors(G,node)]
            
            for neighbour in neighbours:
                N.add_node(tweet_author[neighbour])
                destination = tweet_author[neighbour]
                
                if source == destination:
                    pass
                elif destination in seen:
                    if (destination, source, aggregated_labels[neighbour]) in multi_graph_key_holder[graph_id].keys():
                        U.edges[(destination, source, multi_graph_key_holder[graph_id][(destination, source, aggregated_labels[neighbour])])]["weight"] += 1    
                    else:
                        multi_graph_key_holder[graph_id][(destination, source, aggregated_labels[neighbour])] = index
                        U.add_edge(destination, source, key=index, kind=aggregated_labels[neighbour], weight=1)
                        index += 1

                else:
                    if (source, destination, aggregated_labels[node]) in multi_graph_key_holder[graph_id].keys():
                        U.edges[(source, destination, multi_graph_key_holder[graph_id][(source, destination, aggregated_labels[node])])]["weight"] += 1
                    else:
                        multi_graph_key_holder[graph_id][(source, destination, aggregated_labels[node])] = index
                        U.add_edge(source, destination, key=index, kind=aggregated_labels[node], weight=1)
                        index += 1
                    
                    if (destination, source, aggregated_labels[neighbour]) in multi_graph_key_holder[graph_id].keys():
                        U.edges[(destination, source, multi_graph_key_holder[graph_id][(destination, source, aggregated_labels[neighbour])])]["weight"] += 1
                    else:
                        multi_graph_key_holder[graph_id][(destination, source, aggregated_labels[neighbour])] = index
                        U.add_edge(destination, source, key=index, kind=aggregated_labels[neighbour], weight=1)
                        index += 1
                        
        # We don't add conversations threads (conversations created by just one user)
        if len(U.nodes) > 1:
            viewpoint_networks.append(U)
            viewpoint_network_to_graph_id[U] = graph_id
            conversation_network_to_graph_id[G] = graph_id

    pk.dump(conversation_network_to_graph_id, open(f"{conversation_network_to_graph_id__path_to_save}conversation_network_to_graph_id.pk", "wb"))
            
    return viewpoint_networks, viewpoint_network_to_graph_id, multi_graph_key_holder


def build_viewpoint_matrix(viewpoint_networks_array, viewpoint_network_to_graph_id_dict, key_holder_dict):
    viewpoint_networks = copy.copy(viewpoint_networks_array)
    viewpoint_network_to_graph_id = copy.copy(viewpoint_network_to_graph_id_dict)
    multi_graph_key_holder = copy.copy(key_holder_dict)

    exposed_viewpoints_dict = {}
    exposed_viewpoints_df = {}
    for graph in viewpoint_networks:        
        edges = multi_graph_key_holder[viewpoint_network_to_graph_id[graph]].keys()
        
        exposed_viewpoints_dict[viewpoint_network_to_graph_id[graph]] = {}
        
        for node in graph.nodes:
            influencers = [i for i in nx.Graph.neighbors(graph.reverse(),node)]
            influential_edges = []
            
            for influencer in influencers:
                for edge in edges:
                    if (influencer == edge[0]) and (node == edge[1]):
                        influential_edges.append(edge)
            
            exposed_viewpoints_dict[viewpoint_network_to_graph_id[graph]][node]={label:0 for label in ["L1", "L2", "L3", "L4"]}
            
            for inward in influential_edges:
                exposed_viewpoints_dict[viewpoint_network_to_graph_id[graph]][node][inward[2]] = exposed_viewpoints_dict[viewpoint_network_to_graph_id[graph]][node].get(inward[2], 0) + graph.edges[(inward[0], inward[1], multi_graph_key_holder[viewpoint_network_to_graph_id[graph]][inward] )]["weight"]
        exposed_viewpoints_df[viewpoint_network_to_graph_id[graph]] = pd.DataFrame.from_dict(exposed_viewpoints_dict[viewpoint_network_to_graph_id[graph]])
    return exposed_viewpoints_df


def compute_similiarity(exposed_viewpoints_df):
    exposed_viewpoints = copy.deepcopy(exposed_viewpoints_df)

    cosine_similarity_dict = {}
    cosine_similarity_df = {}

    for graph_id, viewpoint_matrix in exposed_viewpoints.items():
        cosine_similarity_dict[graph_id] = {}
        
        cols = viewpoint_matrix.columns
        df = pd.DataFrame({p:[0 for q in range(len(cols))] for p in cols}, index=[cols])

        pairs = list(itertools.combinations(cols, 2))
        for pair in pairs:
            vector1 = viewpoint_matrix[pair[0]]
            vector2 = viewpoint_matrix[pair[1]]
            cosine_similarity_dict[graph_id][pair] = np.dot(vector1, vector2)/np.round((np.linalg.norm(vector1)*np.linalg.norm(vector2)),10)
            df.at[pair[1], pair[0]] = cosine_similarity_dict[graph_id][pair]
        cosine_similarity_df[graph_id] = df
    return cosine_similarity_dict


def compute_user_level_fragmentation(cosine_similarity_dict):
    cosine_similarity = copy.deepcopy(cosine_similarity_dict)

    user_level_fragmentation = {}
    for graph_id, user_pairs in cosine_similarity.items():
        user_level_fragmentation[graph_id] = {}
        for pair, score in user_pairs.items():
            user_level_fragmentation[graph_id][pair[0]] = user_level_fragmentation[graph_id].get(pair[0], []) + [score]
            user_level_fragmentation[graph_id][pair[1]] = user_level_fragmentation[graph_id].get(pair[1], []) + [score]
        for user_id, user_scores in user_level_fragmentation[graph_id].items():
            user_level_fragmentation[graph_id][user_id] = 1 - np.mean(user_scores)

    user_level_fragmentation_scores = []
    for g_id, pairs in user_level_fragmentation.items():
        user_level_fragmentation_scores += list(pairs.values())

    return user_level_fragmentation_scores


def plot_fragmentation_dist(user_level_fragmentation_scores_array, subject, path="."):
    
    assert subject in ["DST", "Immigration"], "The subject should be either DST or Immigration"

    user_level_fragmentation_scores = copy.deepcopy(user_level_fragmentation_scores_array)

    fig, ax = plt.subplots(figsize=(8,6))
    _ = sns.histplot(user_level_fragmentation_scores, kde=True,bins=20, stat="probability", binrange=[0,1])    
    ax.set_title(subject)
    yticks = np.arange(0,0.9,0.1)
    ax.set_yticks(yticks)
    _ = ax.set_xlabel("Fragmentation score")
    plt.savefig(f"{path}/fragmentation.pdf", bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enter args")
    parser.add_argument('--networks_array_path', help='Enter the path to the networks array', type=str, required=True)
    parser.add_argument('--tweets_authors_path', help='Enter the path to tweet -> author dictionary', type=str, required=True)
    parser.add_argument('--tweets_aggregated_labels_path', help='Enter the path to tweet -> aggregated_labels dictionary', type=str, required=True)
    parser.add_argument('--conversation_network_to_graph_id_path', help='Enter the path to the conversation_network -> graph_id dictionary', type=str, required=True)
    parser.add_argument('--fragmentation_distribution_plot_path', help='Enter the path to save the fragmentation distribution plot', type=str, required=True)
    parser.add_argument('--subject', help='Enter the topic name [DST/Immigration]', type=str, required=True)

    args = parser.parse_args()  

    relevant_network_array = pk.load(open(args.networks_array_path, "rb"))
    tweet_author = pk.load(open(args.tweets_authors_path, "rb"))
    aggregated_labels = pk.load(open(args.tweets_aggregated_labels_path, "rb"))
    conversation_network_to_graph_id_path = args.conversation_network_to_graph_id_path
    fragmentation_distribution_plot_path = args.fragmentation_distribution_plot_path
    subject = args.subject

    viewpoint_networks, viewpoint_network_to_graph_id, multi_graph_key_holder = create_viewpoint_graphs(relevant_network_array, tweet_author, aggregated_labels, conversation_network_to_graph_id_path)
    exposed_viewpoints_df = build_viewpoint_matrix(viewpoint_networks, viewpoint_network_to_graph_id, multi_graph_key_holder)
    cosine_similarity_dict = compute_similiarity(exposed_viewpoints_df)
    user_level_fragmentation_scores = compute_user_level_fragmentation(cosine_similarity_dict)
    plot_fragmentation_dist(user_level_fragmentation_scores, subject, fragmentation_distribution_plot_path)