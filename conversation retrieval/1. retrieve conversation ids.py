import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import requests
import argparse
import os
import json

def read_json(config_file):
	with open(config_file,"r") as f:
		return json.load(f)

def get_convo_id(id_list,config_file):
	x={}
	c=0
	convo = []
	pbar = tqdm(total=len(id_list))
	bearer_tokens = read_json(config_file)
	bearer_tokens = list(bearer_tokens.values())
	index=0
	for i in id_list:
		bearer = bearer_tokens[index]
		pbar.update(1)
		id = str(i)
		url = f"https://api.twitter.com/2/tweets?ids={id}&tweet.fields=conversation_id"
		headers = {"Authorization":"Bearer "+bearer}
		req = requests.get(url,headers=headers)
		x = [req.text]
		convo.append(x)
		if (int(req.headers["x-rate-limit-remaining"]) == 0):
	  		if index+1 < len(bearer_tokens):
	  			index = index+1
  			else:
  				index = 0
  				pbar.set_description("Waiting for Rate Limit")
  				time.sleep(900)

	pbar.close()
	return convo

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Enter args")
	parser.add_argument('--file_name', help='Enter the file name', type=str, required=True)
	parser.add_argument('--sheet_name', help='Enter the sheet name in the file', default='Sheet1', type=str)
	parser.add_argument('--column_name', help='Enter the name of the column that contains the tweet ids', default='ids', type=str)
	parser.add_argument('--config_file', help='Enter the config file name (json format)', default='config.json', type=str)
	args = parser.parse_args()	
	file_name = args.file_name
	sheet_name = args.sheet_name
	column_name = args.column_name
	config_file = args.config_file
	# data = pd.read_excel(file_name,sheet_name)
	data = pd.read_csv(file_name)
	id_list = data[column_name].astype("Int64").to_list()
	conversation_ids = get_convo_id(id_list,config_file)
	np.save("conversation_ids.npy",conversation_ids)