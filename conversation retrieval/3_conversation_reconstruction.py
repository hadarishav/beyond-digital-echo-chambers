import networkx as nx
import numpy as np
from tqdm import tqdm
import copy
import pandas as pd
import pickle as pk
import os


# loading and aggregating the conversations
file = "data_list.npy"
raw_conversations = []
raw_conversations = (np.load(file,allow_pickle=True).tolist())


#Elimiating the NA entries
conversations = [i for i in raw_conversations if i != "NA"]



# This block of code reconstructs Twitter conversations
tweets_array = []
network_array = []
disconnected_tweets = []
disconnected_networks = []
disconnected_networks_nodes = {}
for conversation in tqdm(conversations):
    
    flag1, flag2 = False, False
    network = nx.DiGraph()
    conversation_id = None
    
    if "data" in conversation.keys():
        for tweet in conversation['data']:
            conversation_id = tweet["conversation_id"]
            source = tweet['id']
            if 'referenced_tweets' in tweet.keys():
                for reference in tweet['referenced_tweets']:
                    if reference['type'] == 'replied_to':
                        destination = reference['id']
                        network.add_edge(source, destination)
                        flag1=True
    
    if "includes" in conversation.keys():
        if "tweets" in conversation['includes'].keys():
            for tweet in conversation['includes']['tweets']:
                source = tweet['id']
                if 'referenced_tweets' in tweet.keys():
                    for reference in tweet['referenced_tweets']:
                        if reference['type'] == 'replied_to':
                            destination = reference['id']
                            network.add_edge(source, destination) 
                            flag2=True
    
    if (len(network.nodes) > 0) and (nx.is_connected(network.to_undirected())):
        if len(list(nx.simple_cycles(network)))>0:
            print("Directed cycle found")
        try:
            cycle_found = len(nx.find_cycle(network, orientation="ignore"))>0
            if cycle_found:
                print("Undirected cycle found")
        except:
            pass
    

    if (len(network.nodes) > 0) and (not nx.is_connected(network.to_undirected())):
        nodes = []
        disconnected_tweets.append(conversation)
        disconnected_networks.append(network)
        subgraphs=list(network.subgraph(c) for c in nx.connected_components(network.to_undirected()))
        for component in subgraphs:
            nodes.append(len(list(component.nodes)))
            if conversation_id not in list(component.nodes):
                for node in list(component.nodes):
                    if network.out_degree[node] == 0:
                        network.add_edge(node, conversation_id)
                        break
        disconnected_networks_nodes[len(conversation['data'])] =  disconnected_networks_nodes.get(len(conversation['data']), []) + [nodes]

                        
    if len(network.nodes) > 0 and conversation_id not in list(network.nodes):
        for node in list(network.nodes):
            if network.out_degree[node] == 0:
                network.add_edge(node, conversation_id)
                
                
    
    for node in network.nodes:
        if network.out_degree(node) > 1:
            print("Warning! There is a node with out_degree > 1")
    
    
    if flag1 or flag2:
        tweets_array.append(conversation)
        network_array.append(network)


# Creating one graph consisting of all conversations
hyper_network = nx.DiGraph()
for graph in network_array:
    hyper_network.add_edges_from(graph.edges)     



# Collecting all the tweet texts and authors
tweet_text = {tweet_id:None for tweet_id in nodes}
tweet_author = {tweet_id:None for tweet_id in nodes}
for conversation in tqdm(conversations):
    if "data" in conversation.keys(): 
        for tweet in conversation['data']:
            tweet_text[tweet["id"]] = tweet["text"]
            tweet_author[tweet["id"]] = tweet['author_id']
    if "includes" in conversation.keys():
        if 'tweet' in conversation['includes']:
            for tweet in conversation['includes']['tweets']:
                tweet_text[tweet["id"]] = tweet["text"]
                tweet_author[tweet["id"]] = tweet['author_id']

#I will change the id of tweet_authors with None as user value to a negative number becasue if all the users have None as a username, they will be treated as a same node in the network
number = 0
count = -1
for tweet_id , author_id in tweet_author.items():
    if author_id is None:
        number += 1
        tweet_author[tweet_id] = count
        count -= 1

# Tweets without text
no_text_tweets = {i for i,j in tweet_text.items() if j is None}


#Loading the annotations
annotation_path = ""
annotated_tweets = pd.read_csv(annotation_path)
annotation_dict = pd.DataFrame.to_dict(annotation_df)
for index, relevance in annotation_dict["relevance"].items():
    if relevance == "not english":
        annotation_dict["label"][index] = "L0"
    elif relevance == "no":
        annotation_dict["label"][index] = "L1"
    elif relevance == "yes":
        if annotation_dict["viewpoint"][index] == "none":
            annotation_dict["label"][index] = "L2"
        elif annotation_dict["viewpoint"][index] == "diagnostic":
            annotation_dict["label"][index] = "L3"
        elif annotation_dict["viewpoint"][index] == "counter":
            annotation_dict["label"][index] = "L4"
annotated_tweets_df = pd.DataFrame.from_dict(annotation_dict)
annotated_tweets_df = annotated_tweets_df[["tweet_id", "label"]]
annotated_tweets_df.index = annotated_tweets_df["tweet_id"]
labels_dict = pd.DataFrame.to_dict(annotated_tweets_df)["label"]

# Collecting non-English tweets
no_englis_labels = {i for i,j in labels_dict.items() if j == "L0"}

# Aggregating the labels
aggregated_labels = {}
for tweet_id, label in labels_dict.items():
    if label != "L0":
        aggregated_labels[tweet_id] = label


# Removing non-English nodes and the ones without text from the hypergraph consisting of all conversations
to_be_removed_tweets = no_text_tweets.union(no_englis_labels)
modified_network_array = []
zeros, ones = [], []
for network in tqdm(network_array): 
    flag = True
    under_operation_network = copy.deepcopy(network)
    nodes_list = network.nodes
    
    for node in nodes_list:
        if node in to_be_removed_tweets:
            if under_operation_network.out_degree(node) == 0:
                if under_operation_network.in_degree(node) <= 1:
                    under_operation_network.remove_node(node)
                elif under_operation_network.in_degree(node) > 1:
                    in_edges = list(under_operation_network.in_edges(node))
                    new_root = in_edges[0][0]
                    for elm in in_edges:
                        if new_root not in elm: 
                            under_operation_network.add_edge(elm[0], new_root)
                    under_operation_network.remove_node(node)
                            
            elif under_operation_network.out_degree(node) == 1:
                if under_operation_network.in_degree(node) == 0:
                    under_operation_network.remove_node(node)
                elif under_operation_network.in_degree(node) >= 1:
                    in_edges = list(under_operation_network.in_edges(node))
                    for elm in in_edges:
                        under_operation_network.add_edge(elm[0], list(under_operation_network.out_edges(node))[0][1])
                    under_operation_network.remove_node(node)
                    
    net_size = len(under_operation_network.nodes) 
    if net_size == 0:
        zeros.append(under_operation_network)
        flag = False
    elif net_size == 1:
        ones.append(under_operation_network)
        flag = False
        
    if (len(under_operation_network.nodes) > 0) and (nx.is_connected(under_operation_network.to_undirected())):
        if len(list(nx.simple_cycles(under_operation_network)))>0:
            print("Directed cycle found")
        try:
            cycle_found = len(nx.find_cycle(under_operation_network, orientation="ignore"))>0
            if cycle_found:
                print("Undirected cycle found")
        except:
            pass
    
    if (len(under_operation_network.nodes) > 0) and (not nx.is_connected(under_operation_network.to_undirected())):
        print("the network is disconnected")                        
                
    for node in under_operation_network.nodes:
        if under_operation_network.out_degree(node) > 1:
            print("Warning! node with out-degree>1 is found")   
    
    if flag:
        modified_network_array.append(under_operation_network)
modified_hyper_network = nx.DiGraph()
for graph in modified_network_array:
    modified_hyper_network.add_edges_from(graph.edges)


# Adding tweet text and annotation to the modified network nodes
for network in modified_network_array:
    for node in network.nodes:
        network.nodes[node]["text"] = tweet_text[node]
        network.nodes[node]["annotation"] = aggregated_labels[node]
subgraphs=list(modified_hyper_network.subgraph(c) for c in nx.connected_components(modified_hyper_network.to_undirected()))
for component in subgraphs:
    for node in list(component.nodes):
        component.nodes[node]["text"] = tweet_text[node]
        component.nodes[node]["annotation"] = aggregated_labels[node]


# In this part conversations which all their nodes are irrelevant are removed 
relevant_network_array = []
for network in modified_network_array:
    flag = False
    for node in network.nodes:
        if network.nodes[node]["annotation"] != "L1":
            flag = True
            break
    if flag:
        relevant_network_array.append(network)
relevant_modified_hyper_network_with_attributes = nx.DiGraph()
for graph in relevant_network_array:
    relevant_modified_hyper_network_with_attributes.add_nodes_from(graph.nodes)
    relevant_modified_hyper_network_with_attributes.add_edges_from(graph.edges)
subgraphs=list(relevant_modified_hyper_network_with_attributes.subgraph(c) for c in nx.connected_components(relevant_modified_hyper_network_with_attributes.to_undirected()))
for component in subgraphs:
    for node in list(component.nodes):
        component.nodes[node]["text"] = tweet_text[node]
        component.nodes[node]["annotation"] = aggregated_labels[node]

pk.dump(relevant_network_array, open("relevant_network_array.pk", "wb"))
pk.dump(relevant_modified_hyper_network_with_attributes, open("conversations_relevant_modified_with_attributes.pk", "wb"))

