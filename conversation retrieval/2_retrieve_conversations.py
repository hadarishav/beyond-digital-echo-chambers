import requests
import json
import time
import ast
import numpy as np
from tqdm import tqdm
import argparse

search_url = "https://api.twitter.com/2/tweets/search/all"
def read_json(config_file):
	with open(config_file,"r") as f:
		return json.load(f)

def create_convo_id_dict(conversations_id_list):
	convo_id_dict = {}
	convo_id_list = []
	for i in conversations_id_list:
		res = ast.literal_eval(i[0])
		if "data" in res:
			c_id = res["data"][0]["conversation_id"]
			id = res["data"][0]["id"]
		else:
			c_id = "NA"  #FOR SOME IDS THERE IS NO CONVERSATION ID
			try:
				id = res["errors"][0]["resource_id"]
			except:
				id = "none"
		convo_id_dict[id] = c_id
	convo_id_list = list(set(convo_id_dict.values()))
	return convo_id_dict,convo_id_list

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

def connect_to_endpoint(url, headers, params):
    response = requests.request("GET",
                                search_url,
                                headers=headers,
                                params=params)
    if response.status_code == 200:
      return response.status_code, response.json()
    else:
      return response.status_code, []

def func_call(id,query_params,bearer):
    headers = create_headers(bearer)
    status_code, json_response = connect_to_endpoint(search_url, headers, query_params)
    return status_code, json_response

def get_convo(conversation_id,query_params,bearer):
	status_code, json_response = func_call(conversation_id,query_params,bearer)
	return status_code, json_response

def id_iter(convo_id_dict,config_file):
	# convo_id_dict = ["1255183480859824129"]
	data_list = []
	pbar = tqdm(total=len(convo_id_dict))
	bearer_tokens = read_json(config_file)
	bearer_tokens = list(bearer_tokens.values())
	index=0
	ctr=0
	for conversation_id in convo_id_dict:
		ctr+=1
		pbar.update(1)
		bearer = bearer_tokens[index]
		if conversation_id!= "NA":
			query_params = {
			'query': 'conversation_id:{}'.format(conversation_id),
			'tweet.fields': 'conversation_id,created_at',
			'expansions' : 'author_id,referenced_tweets.id',
			'user.fields' : 'username',
			'start_time': '2018-01-25T00:00:00Z',
			'max_results': 50}
			status_code, json_response = get_convo(conversation_id,query_params,bearer)
			if status_code == 429:
				if index+1 < len(bearer_tokens):
					index = index+1
				else:
					index=0
					pbar.set_description("Waiting for Rate Limit")
					time.sleep(900)
				bearer = bearer_tokens[index]
				status_code, json_response = get_convo(conversation_id,query_params,bearer)
				time.sleep(1)
			data_list.append(json_response)
		else:
			data_list.append("NA")
		time.sleep(1)
	pbar.close()
	return data_list

if __name__ == "__main__":

	print("This code does not work with Standard API access")
	parser = argparse.ArgumentParser(description="Enter args")
	parser.add_argument('--config_file', help='Enter the config file name (json format)', default='config.json', type=str)
	args = parser.parse_args()	
	config_file = args.config_file
	conversations_id_list = np.load("conversation_ids.npy",allow_pickle=True)
	convo_id_dict,convo_id_list = create_convo_id_dict(conversations_id_list)
	data_list = id_iter(convo_id_list,config_file)
	np.save("data_list.npy",data_list)