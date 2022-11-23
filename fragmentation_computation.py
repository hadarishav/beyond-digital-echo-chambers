import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import seaborn as sns



# In this section, we create viewpoint graphs
relevant_network_array = pk.load(open("relevant_network_array.pk", "rb"))
conversation_viewpoint = {}
viewpoint_networks = []
multi_graph_key_holder = {}
viewpoint_network_to_graph_id = {}
graph_id_to_viewpoint_network = {}
node_collection = {}

conversation_network_to_graph_id = {}
graph_id_to_conversation_network = {}
binary_user_conversations = 0

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
        graph_id_to_viewpoint_network[graph_id] = U
        conversation_viewpoint[G] = U

        conversation_network_to_graph_id[G] = graph_id
        graph_id_to_conversation_network[graph_id] = G
    node_collection[graph_id] = N
        
    if len(U.nodes) == 1:
        print(U.nodes)
        print(U.edges)
        
    if len(U.nodes) == 2:
        binary_user_conversations += 1


pk.dump(conversation_network_to_graph_id, open("conversation_network_to_graph_id.pk", "wb"))


# In this section, the viewpoint matrix is created
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


# Measuring the similarity between pairs of user-viewpoint vectors
cosine_similarity_dict = {}
cosine_similarity_df = {}

for graph_id, viewpoint_matrix in exposed_viewpoints_df.items():
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

user_level_fragmentation = {}
for graph_id, user_pairs in cosine_similarity_dict.items():
    user_level_fragmentation[graph_id] = {}
    for pair, score in user_pairs.items():
        user_level_fragmentation[graph_id][pair[0]] = user_level_fragmentation[graph_id].get(pair[0], []) + [score]
        user_level_fragmentation[graph_id][pair[1]] = user_level_fragmentation[graph_id].get(pair[1], []) + [score]
    for user_id, user_scores in user_level_fragmentation[graph_id].items():
        user_level_fragmentation[graph_id][user_id] = 1 - np.mean(user_scores)

user_level_fragmentation_scores = []
for g_id, pairs in user_level_fragmentation.items():
    user_level_fragmentation_scores += list(pairs.values())

fig, ax = plt.subplots(figsize=(8,6))
sns.histplot(user_level_fragmentation_scores, kde=True,bins=20, stat="probability", binrange=[0,1])
# ax.get_legend().remove()
if subject=="IMMIGRATION":
    ax.set_title("Immigration")
elif subject=="DTS":
    ax.set_title("DST")
    
ax.set_ylim(0,0.7)
ax.set_xlabel("Fragmentation score")
plt.savefig(".fragmentation.pdf", bbox_inches='tight')