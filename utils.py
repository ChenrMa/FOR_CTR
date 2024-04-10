import os
import pandas as pd
import random
from collections import defaultdict
import torch
import numpy as np


#only test
def load_data(dataset):
    video_dir = os.path.join('video',dataset)
    data = pd.read_excel('dataset/video.xlsx', index_col='col_name')
    files = os.listdir(video_dir)
    lens = len(files)
    train_score = {}
    valid_score = {}
    test_score = {}
    train = int(lens*0.6)
    valid = int(lens*0.8)
    for i in range(train):
        for j in range(i + 1,train):
            CTR1 = data.loc[files[i], dataset]
            CTR2 = data.loc[files[j], dataset]
            if CTR1 > CTR2:
                train_score[files[i],files[j]] = 0
            elif CTR1 < CTR2:
                train_score[files[i],files[j]] = 1
    for i in range(train,valid):
        for j in range(i + 1,valid):
            CTR1 = data.loc[files[i], dataset]
            CTR2 = data.loc[files[j], dataset]
            if CTR1 > CTR2:
                valid_score[files[i],files[j]] = 0
            elif CTR1 < CTR2:
                valid_score[files[i],files[j]] = 1
    for i in range(valid,lens):
        for j in range(i + 1,lens):
            CTR1 = data.loc[files[i], dataset]
            CTR2 = data.loc[files[j], dataset]
            if CTR1 > CTR2:
                test_score[files[i],files[j]] = 0
            elif CTR1 < CTR2:
                test_score[files[i],files[j]] = 1
    return train_score, valid_score, test_score

def build_video_dict(dataset):
    video_dir = os.path.join('video',dataset)
    files = os.listdir(video_dir)
    video_dict = {}
    nums = 0
    for i in files:
        video_dict[i] = nums
        nums+=1
    train = [i for i in list(video_dict)[:int(len(files)*0.6)]]
    valid = [i for i in list(video_dict)[:int(len(files) * 0.8)]]
    test = [i for i in list(video_dict)]
    return video_dict,train,valid,test

def get_key (dict, value):
    return [k for k, v in dict.items() if v == value]


def five_level(train,dataset):
    all_score = {}
    data = pd.read_excel('dataset/video.xlsx', index_col='col_name')
    for i in train:
        all_score[i] = data.loc[i, dataset]
    V_K = {v: k for k, v in all_score.items()}
    sort_V_K = sorted(V_K.keys())
    final = [[]for i in range(5)]
    for nums in range(5):
        for i in sort_V_K[int(len(sort_V_K)*0.2*nums):int(len(sort_V_K)*0.2*(nums+1))]:
            for j in get_key(all_score, i):
                final[nums].append(j)
    labels = {}
    for i in range(5):
        for j in final[i]:
            labels[j] = i
    return labels
    
def va_loss(score, true_value):
    loss = []
    for i in true_value:
        res = true_value[i]
        pos = 0-int(res)
        neg = 1-int(res)
        if score[i[0]] > score[i[1]]:
            predict = 0
        else:
            predict = 1
        if true_value[i] != predict:
            loss.append(-torch.log(1e-8 + torch.sigmoid(score[i[pos]]))-torch.log(1e-8 + (1 - torch.sigmoid(score[i[neg]]))))
    avg_loss = torch.mean(torch.stack(loss))
    return avg_loss

def evaluate(score, true_value):
    hit = 0
    for i in true_value:
        if score[i[0]] > score[i[1]]:
            predict = 0
        else:
            predict = 1
        if true_value[i] == predict:
            hit+=1
    hit = hit/len(true_value)
    return hit

def test_scores(test,dataset):
    data = pd.read_excel('dataset/video.xlsx', index_col='col_name')
    ctrs = {}
    for i in test:
        CTR = data.loc[i, dataset]
        ctrs[i] = CTR
    return ctrs

def _rank_correlation_(att_map, att_gd):
        n = torch.tensor(att_map.shape[1])
        upper = 6 * torch.sum((att_gd - att_map).pow(2), dim=1)
        down = n * (n.pow(2) - 1.0)
        return (1.0 - (upper / down)).mean(dim=-1)

def Spearmans_Correlation(test,score,dataset):
    ctrs = test_scores(test,dataset)
    a = sorted(ctrs.items(), key=lambda x: x[1],reverse=True)
    rel = [i for i in range(1,len(a)+1)]
    dic = {}
    num = 0
    for i in a:
        dic[i[0]] = rel[num]
        num+=1
    rel = torch.tensor([rel])
    pred = sorted(score.items(), key=lambda x: x[1],reverse=True)
    pre = []
    for i in pred:
        pre.append(dic[i[0]])
    pre = torch.tensor([pre])
    att = rel.sort(dim=1)[1]
    grad_att = pre.sort(dim=1)[1]
    correlation = _rank_correlation_(att.float(), grad_att.float())
    return correlation

def val_NDCG(test,score,dataset):
    def getDCG(scores):
        return np.sum(
                np.divide(scores, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)+1),
                dtype=np.float32)


    def getNDCG(rank_list, pos_items):
        relevance = [i for i in np.arange(len(pos_items),0,-1)]
        it2rel = {it: r for it, r in zip(pos_items, relevance)}
        rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)

        idcg = getDCG(np.sort(relevance)[::-1])

        dcg = getDCG(rank_scores)

        if dcg == 0.0:
            return 0.0
        print(dcg,idcg)
        ndcg = dcg / idcg
        return ndcg
    ctrs = test_scores(test,dataset)
    a = sorted(ctrs.items(), key=lambda x: x[1],reverse=True)
    rel = []
    for i in a:
        rel.append(i[0])
    pred = sorted(score.items(), key=lambda x: x[1],reverse=True)
    pre = []
    for i in pred:
        pre.append(i[0])
    return getNDCG(pre, rel)


