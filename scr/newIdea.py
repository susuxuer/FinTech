#!/usr/bin/python
#-*- coding:UTF-8-*-

import jieba
import jieba.posseg as pseg                 #引入结巴分词词性标注
import jieba.analyse
import numpy as np
import pandas
import pandas as pd
import csv
from gensim import corpora,models,similarities      #引入文本相似度库
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from pandas import DataFrame

import time

#=============================================训练集分词＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
#读取文件，主要用以生成词库
def loadPoorEnt(path2 = './stopwords.tab'):

    poor_ent=set([])
    with open(path2, 'r') as ropen:
        lines=ropen.readlines()
        for line in lines:
            line=line.replace('\r','').replace('\n','')
            poor_ent.add(line)
    return poor_ent

stop_words=loadPoorEnt()

#读取数据
def extract_data(path,news_line):
    trainFile = pd.read_csv(path)  # 训练集文件
    trainFile=trainFile[:news_line]
    id=trainFile[['id']]
    title=trainFile[['title']]
    return id.values.tolist(),title.values.tolist()

#分词
def cut(data):
    result=[]    #pos=['n','v']
    for line in data:
        line=line[0]
        res=pseg.cut(line.strip())
        list=[]
        for item in res:
            if len(item.word)>1 and item.word.encode('utf8') not in stop_words and (item.flag.startswith(u'n') or item.flag.startswith(u'v')):
            #if item.word not in stop_words and (item.flag.startswith('n')):
                list.append(item.word)
        result.append(list)
    return result

#提取重要的特征(TF-IDF) 方法一：（每条新闻中tfidf排在top的词语）
def extract_features(data,top=3):
    dict=Dictionary(data)
    print len(dict)
    corpus = [dict.doc2bow(line) for line in data]
    model = TfidfModel(corpus)
    #每个词取出前top个特征（tfidf最大的top个）
    features=set([])
    for item in corpus:
        vector=model[item]
        vector_sorted=sorted(vector,key=lambda x:x[1],reverse=True)
        for item in vector_sorted[:top]:
            features.add(dict[item[0]])

    print len(features)
    print ' '.join(list(features))
    return features

#提取重要的特征(TF-IDF) 方法二：（TF-IDF累加）
def calculate_tfidf_scores(top,news_length):
    id,docs=extract_data('./train_data.csv',news_length)
    data=cut(docs)
    dict=Dictionary(data)
    print len(dict)
    corpus = [dict.doc2bow(line) for line in data]
    model = TfidfModel(corpus)
    map={}
    for cor in corpus:
        tfidf=model[cor]
        for item in tfidf:
            id=item[0]
            value=item[1]
            temp=map.get(id,None)
            if temp==None:
                map[id]=value
            else:
                map[id]+=temp
    result=sorted(map.items(),key=lambda x:x[1],reverse=True)
    max_value_feat=[]
    for id in result[:top]:
        word=dict[id[0]]
        max_value_feat.append(word)
        print word
    return max_value_feat
    # all_corpus=[]
    # for item in corpus:
    #     vector=model[item]
    #     for word in item:
    #         for j in range(len(corpus)):
    #             for m in range(len(corpus[j])):
    #                 if word[0] == corpus[j][m][0]:
    #                     word[1]+=corpus[j][m][1]
    #         all_corpus.append(word)
    # print all_corpus
    print model.dfs
    print model.idfs
#重新计算tfidf
def cal_tfidf(data,feature):
    result=[]
    #pos=['n','v']
    for line in data:
        line=line[0]
        res=pseg.cut(line.strip())
        list=[]
        for item in res:
            if item.word in feature:
                list.append(item.word)
        result.append(list)
    data=result
    dict = Dictionary(data)
    corpus = [dict.doc2bow(line) for line in data]
    model = TfidfModel(corpus)
    return model,dict,corpus,feature

#用于将词汇向量化
def list2vec(list,length):
    vector=np.zeros(length)
    for item in list:
        vector[item[0]]=item[1]
    return vector

#48W文本向量化并保存
def text2vec(path,news_length):
    id,docs=extract_data('./train_data.csv',news_length)
    data_cut=cut(docs)
    feature=extract_features(data_cut)
    model, dict, corpus, feature = cal_tfidf(docs,feature)
    length = len(feature)
    text_vec =[]
    for i,corp in enumerate(corpus):
        comp_vec=model[corp]
        vec_each=list2vec(comp_vec,length)
        text_vec.append(vec_each)
    # df = DataFrame(text_vec[0])
    # df.to_csv('./result.txt', sep='\t', index=False)
    #print type(text_vec), len(text_vec)
    return text_vec
#得到新闻的每条句子
def get_sentence():
    pass

def similar(docs,model,dict,corpus,feature,id,i):
    result=[]
    #将sentence转化为向量
    # sen=[]
    # res=pseg.cut(sentence)
    # for item in res:
    #     if item.word in feature:
    #         sen.append(item.word)
    #其实就是语料库中前20条的向量
    tfidf=model[corpus[i]]
    length=len(feature)
    vec_a=list2vec(tfidf,length)
    #text_vec=list2vec()
    # for j in range(5):
    #     vec_a =  corpus[j]
    for i,corp in enumerate(corpus):
        comp_vec=model[corp]
        vec_b=list2vec(comp_vec,length)
        num=np.dot(vec_a,vec_b.T)
        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        cos = num / denom  # 余弦值
        result.append((cos,i))
    #对所有的余弦进行排序，给出前20条序号即可
    result=sorted(result,key=lambda x:x[0],reverse=True)
#相似的文档
    sim_id=[]
    for item in result[:40]:
        sim_id.append(id[item[1]][0])
        print id[item[1]][0]
        print docs[item[1]][0]
    return sim_id

def synthesis():
    id,docs=extract_data('./train_data.csv',10000)
    data_cut=cut(docs)
    feature=extract_features(data_cut)
    model, dict, corpus, feature=cal_tfidf(docs,feature)
    all_sim_id=[]
    for i in range(3):
        #similar(docs,model,dict,corpus,feature,id,i)
        sim_id = similar(docs, model, dict, corpus, feature, id, i)
        all_sim_id.append(sim_id)
    return all_sim_id

#输出最终的解
def out_txt(test_id,sim_id,title):
    result=[]
    for i,sim in enumerate(sim_id):
        test=test_id[i]
        for j in range(1,21):
            result.append([test,sim[j],])
    df=DataFrame(result,columns=['source_id','target_id'])
    df.to_csv('./result.txt',sep='\t',index=False)

def cal_time(time):
    if time<60:
        return str(time)+' secs '
    elif time<60*60:
        return str(time/(60.0))+' mins '
    else:
        return str(time/(60*60.0))+' hours '


if __name__=='__main__':
   #  test_id, docs = extract_data('./test_data.csv',50)
   #  all_test_id = [test_id[i][0] for i in range(50)]
   #  int_id= [int(i) for i in all_test_id]
   #  int_test_id=[]
   #  for i in range(50):
   #       int_test_id.append(int_id[i])
   # # print int_test_id
   #  all_sim_id = synthesis()
   #  out_txt(int_test_id, all_sim_id,docs)

    # for i in range(1,21):
    #     out_txt(test_id,all_sim_id)

    start_time=time.time()
    calculate_tfidf_scores(5000,50000)
    end_time = time.time()
    print 'elapse time',cal_time(end_time-start_time)
