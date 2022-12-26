import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from scipy.special import kl_div as kl
import argparse

def per_convo_distribution(x):
    all_labels=[]
    count_convo={}
    for i in x:
        all_labels.extend(x[i])
        convo_len = len(x[i])
        count_convo[i]= Counter(x[i])
        if ("L1" not in count_convo[i]):
            count_convo[i]["L1"]=0
        if ("L2" not in count_convo[i]):
            count_convo[i]["L2"]=0
        if ("L3" not in count_convo[i]):
            count_convo[i]["L3"]=0
        if ("L4" not in count_convo[i]):
            count_convo[i]["L4"]=0
        for j in count_convo[i]:
            count_convo[i][j]=count_convo[i][j]/convo_len
    
    convo_dist=[]
    for i in count_convo:
        convo_dist.append([count_convo[i]["L1"],count_convo[i]["L2"],count_convo[i]["L3"],count_convo[i]["L4"]])

    return all_labels, convo_dist


def overall_distribution(all_labels):
    main_count = Counter(all_labels)
    for i in main_count:
        main_count[i] = main_count[i]/len(all_labels)
    main_count_list = [main_count['L1'],main_count['L2'],main_count['L3'],main_count['L4']]
    return main_count_list

  
def KL_div(convo_dist):
    kl_list1=[]
    for i in convo_dist:
        kl_list1.append(sum(kl(i,main_count_list)))
    return kl_list1
  

def normalize_KL(kl_list1):
    max_kl_list1 = max(kl_list1)
    kl_list1 = [i/max_kl_list1 for i in kl_list1]
    return kl_list1

def plot_fig(kl_list1,topic):
    sns.set_theme()
    sns.set(font_scale=2)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(kl_list1,kde=True,stat="probability",bins=20).set(xlabel="Representation score",title=topic,ylim=(0,0.7))
    plt.savefig("rep.pdf", bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enter args")
    parser.add_argument('--file_name', help='Enter the file name', type=str, required=True)
    parser.add_argument('--topic', help='Enter the topic name [DST/Immigration]', type=str, required=True)
    args = parser.parse_args()	
    file_name = args.file_name
    topic = args.topic
    conversation_list=pickle.load(open(file_name, "rb")) #load a list of conversations, where each conversation is a list of viewpoint label Ex. [[L1,L1,L2,L4],....,[L4,L3,L2,L1,L3]]
    all_labels, convo_dist = per_convo_distribution(conversation_list)
    main_count_list = overall_distribution(all_labels)
    kl_list1 = KL_div(convo_dist)
    kl_list1 = normalize_KL(kl_list1)
    plot_fig(kl_list1,topic)
    