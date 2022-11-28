from anytree import Node, RenderTree
from collections import defaultdict 
from anytree import Node, RenderTree, AsciiStyle, PreOrderIter
import time
import tweepy
import pandas as pd
import numpy as np


def create_dicts(data_list):
	username_dict={}
	tweet_dict={}
	author_id_dict={}
	ref_id_dict={}
	for data in data_list:
		if 'data' in data:
			# print(data)
			# print("hello")
			user_list=data["includes"]["users"]
			for user in user_list:
				username_dict[user["id"]]=user["username"]
			username_dict["NA"] = "Not found error"
			for i in data["data"]:
				tweet_dict[i["id"]] = i["text"] 
				author_id_dict[i["id"]]=i["author_id"]
				for j in i["referenced_tweets"]:
					if j["type"] == "replied_to":
						ref_id_dict[i["id"]] = j["id"]
			if "tweets" in data["includes"]:
				includes_tweets_list = data["includes"]["tweets"]
				for tweet in includes_tweets_list:
					tweet_dict[tweet["id"]]=tweet["text"]
					author_id_dict[tweet["id"]]=tweet["author_id"]
					if "referenced_tweets" in tweet:
						for j in tweet["referenced_tweets"]:
							if j["type"] == "replied_to":
								ref_id_dict[tweet["id"]] = j["id"]
			if "errors" in data["includes"]:
				includes_tweets_list = data["includes"]["errors"]
				for tweet in includes_tweets_list:
					tweet_dict[tweet["value"]]="Not found error"
					author_id_dict[tweet["value"]]="NA"

	return username_dict,tweet_dict,author_id_dict,ref_id_dict

def create_trees(data_list):
	num_tweets=0
	c=0
	tree = {} 
	for x in data_list:
		if 'data' in x:
			for i in x['data']:
				c+=1
				id = str(i["id"])
				for j in i["referenced_tweets"]:
					if j["type"] == "replied_to":
						reply_id = j["id"]
				if (reply_id not in tree):
					tree[reply_id] = Node(reply_id)
				if id in tree:
					tree[id].parent = tree[reply_id]
				else:
					tree[id] = Node(id, parent = tree[reply_id])
	return tree

def create_tree_order(data_list):
	convo_list=[]
	for i in data_list:
		if "data" in i:
			c = i["data"][0]["conversation_id"]
			if i["meta"]["result_count"] == 1:
				if i["data"][0]["id"] != i["data"][0]["conversation_id"]:
					c = i["data"][0]["id"]
			try:
				a = [node.name for node in PreOrderIter(tree[c])]
			except:
				tree[c] = Node(c)
				for tweet in i["data"]:
					id = tweet["id"]
					tree[id].parent = tree[c]
				a = [node.name for node in PreOrderIter(tree[c])]
			convo_list.append(a)
	return convo_list

def create_file(convo_list,username_dict,tweet_dict,author_id_dict,ref_id_dict):
	tweet_id = []
	text = []
	reply_to = []
	username = []
	tweet_level = []
	previous_tweet=[]
	c=0
	for i in convo_list:
		for j in i:
			tweet_id.append("ID_" + str(j))
			if j in tweet_dict:
				text.append(tweet_dict[j].strip())
			else:
				text.append("NA")	
			if j in ref_id_dict:
				rep_id = ref_id_dict[j]
			else:
				rep_id = "NA"
			reply_to.append("ID_" + str(rep_id))
			if j in author_id_dict:
				if author_id_dict[j] in username_dict:
					username.append(username_dict[author_id_dict[j]])
				else:
					username.append("USER_ID_"+ author_id_dict[j])
			else:
				username.append("NA")
			if len(i) == 1 and len(tree[j].ancestors) == 1:
				tweet_level.append("1_root")
			else:
				tweet_level.append(len(tree[j].ancestors))
			if rep_id in tweet_dict:
				previous_tweet.append(tweet_dict[rep_id])
			else:
				previous_tweet.append("NA")
			c+=1
	text = [k.replace("\n", " ").replace("\r", "") for k in text]
	previous_tweet = [k.replace("\n", " ").replace("\r", "") for k in previous_tweet]
	return tweet_id,text,reply_to,username,tweet_level,previous_tweet


if __name__ == "__main__":

	data_list = np.load("data_list.npy",allow_pickle=True).tolist()
	username_dict,tweet_dict,author_id_dict,ref_id_dict = create_dicts(data_list)
	tree = create_trees(data_list)
	convo_list = create_tree_order(data_list)
	tweet_id,text,reply_to,username,tweet_level,previous_tweet = create_file(convo_list,username_dict,tweet_dict,author_id_dict,ref_id_dict)
	col_names = ["tweet_id","text", "reply_to","username","tweet_level","previous_tweet"]
	format = pd.DataFrame(columns = col_names)
	format.tweet_id = tweet_id
	format.text = text
	format.reply_to=reply_to
	format.username = username
	format.tweet_level = tweet_level
	format.previous_tweet = previous_tweet
	format.to_csv("conversations.csv")