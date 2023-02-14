import networkx as nx
import numpy as np
from tqdm import tqdm
import copy
import pandas as pd
import pickle as pk
import os


# loading and aggregating the conversations
def load_raw_conversations(path):
    filenames = [i for i in os.listdir(path)]
    raw_conversations = []
    for filename in filenames:
        raw_files = (np.load(f"{path}{filename}",allow_pickle=True).tolist())
        raw_conversations += raw_files

    #Elimiating the NA entries
    conversations = [i for i in raw_conversations if i != "NA"]

    return conversations


# This block of code reconstructs Twitter conversations
def reconstruct_conversations(conversations):
    all_conversations = copy.deepcopy(conversations)
    network_array = []
    for conversation in tqdm(all_conversations):
        
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
            subgraphs=list(network.subgraph(c) for c in nx.connected_components(network.to_undirected()))
            for component in subgraphs:
                nodes.append(len(list(component.nodes)))
                if conversation_id not in list(component.nodes):
                    for node in list(component.nodes):
                        if network.out_degree[node] == 0:
                            network.add_edge(node, conversation_id)
                            break

                            
        if len(network.nodes) > 0 and conversation_id not in list(network.nodes):
            for node in list(network.nodes):
                if network.out_degree[node] == 0:
                    network.add_edge(node, conversation_id)
                    
        for node in network.nodes:
            if network.out_degree(node) > 1:
                print("Warning! There is a node with out_degree > 1")
        
        if flag1 or flag2:
            network_array.append(network)
            
    nodes = set()
    for net in network_array:
        for node in list(net.nodes):
            nodes.add(node)

    return network_array, nodes


# Collecting all the tweet texts and authors
def retrieve_tweet_text_and_tweet_author(path, conversations, nodes):
    all_conversations = copy.deepcopy(conversations)
    all_nodes = copy.deepcopy(nodes)
    tweet_text = {tweet_id:None for tweet_id in all_nodes}
    tweet_author = {tweet_id:None for tweet_id in all_nodes}
    for conversation in tqdm(all_conversations):
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

    pk.dump(tweet_author, open(f"{path}tweet_author.pk", "wb"))
    pk.dump(tweet_text, open(f"{path}tweet_text.pk", "wb"))

    return tweet_author, tweet_text


#Loading the annotations
def load_labels(loading_path, saving_path, tweet_text_dict):
    tweet_text = copy.deepcopy(tweet_text_dict)
    
    annotation_path = loading_path
    annotated_tweets = pd.read_csv(annotation_path)
    
    no_text_tweets = {i for i,j in tweet_text.items() if j is None}
    
    annotated_df = annotated_tweets[['tweet_id', "relevance", "viewpoint"]]
    annotated_df['tweet_id'] = annotated_df['tweet_id'].astype('str')
    annotation_df = annotated_df.loc[~annotated_df['tweet_id'].isin(no_text_tweets)]
    for col in ["relevance", "viewpoint"]:
        annotation_df.loc[:][col] = annotation_df.loc[:][col].str.strip()
        
    annotation_dict = pd.DataFrame.to_dict(annotation_df)
    annotation_dict["label"] = {i:None for i in annotation_dict["tweet_id"].keys()}
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


    # Aggregating the labels
    aggregated_labels = {}
    for tweet_id, label in labels_dict.items():
        if label != "L0":
            aggregated_labels[tweet_id] = label

    pk.dump(aggregated_labels, open(f"{saving_path}aggregated_labels.pk", "wb"))

    return labels_dict, aggregated_labels


def build_networks_array(tweet_text_dict, annotation_dict, aggregated_annotation_dict, networks_array, path):
    tweet_text = copy.deepcopy(tweet_text_dict)
    labels_dict = copy.deepcopy(annotation_dict)
    aggregated_labels = copy.deepcopy(aggregated_annotation_dict)
    network_array = copy.deepcopy(networks_array)

    # Tweets without text
    no_text_tweets = {i for i,j in tweet_text.items() if j is None}

    # Collecting non-English tweets
    no_englis_labels = {i for i,j in labels_dict.items() if j == "L0"}

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


    # Adding tweet text and annotation to the modified network nodes
    for network in modified_network_array:
        for node in network.nodes:
            network.nodes[node]["text"] = tweet_text[node]
            network.nodes[node]["annotation"] = aggregated_labels[node]

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

    pk.dump(relevant_network_array, open(f"{path}relevant_network_array.pk", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enter args")
    parser.add_argument('--raw_conversations_path', help='Enter the path to the raw conversations', type=str, required=True)
    parser.add_argument('--tweet_author_and_tweet_text_saving_path', help='Enter the path to save tweet_author and tweet_text dictionaries', type=str, required=True)
    parser.add_argument('--annotation_loading_path', help='Enter the path to load the annotation file', type=str, required=True)
    parser.add_argument('--annotation_saving_path', help='Enter the path to save the aggregated annotation file', type=str, required=True)
    parser.add_argument('--networks_array_path', help='Enter the path to save networks array', type=str, required=True)

    args = parser.parse_args()  

    raw_conversations_path = args.raw_conversations_path
    tweet_author_and_tweet_text_saving_path = args.tweet_author_and_tweet_text_saving_path
    annotation_loading_path = args.annotation_loading_path
    annotation_saving_path = args.annotation_saving_path
    networks_array_path = args.networks_array_path

    conversations = load_raw_conversations(raw_conversations_path)
    network_array, nodes = reconstruct_conversations(conversations)
    tweet_author, tweet_text = retrieve_tweet_text_and_tweet_author(tweet_author_and_tweet_text_saving_path, conversations, nodes)
    labels_dict, aggregated_labels = load_labels(annotation_loading_path, annotation_saving_path, tweet_text)
    build_networks_array(tweet_text, labels_dict, aggregated_labels, network_array, networks_array_path)