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
from collections import defaultdict
import time

#=============================================训练集分词＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
#读取文件，主要用以生成词库
def loadPoorEnt(path2 = './stopwords_suxue.tab'):

    poor_ent=set([])
    with open(path2, 'r') as ropen:
        lines=ropen.readlines()
        for line in lines:
            line=line.replace('\r','').replace('\n','')
            poor_ent.add(line)
    return poor_ent
stop_words=loadPoorEnt()

#读取数据
def extract_data(path,top):
    trainFile = pd.read_csv(path)  # 训练集文件
    trainFile=trainFile[:top]
    id=trainFile[['id']]
    title=trainFile[['title']]
    return id.values.tolist(),title.values.tolist()

#分词
def cut(data):
    result=[]    #pos=['n','v']
    for line in data:
        line = line[0]
        res = pseg.cut(line)
        list = []
        for item in res:
            if item.word.encode('utf8') not in stop_words :
                list.append(item.word)
                #print item.word
        result.append(list)
        #print "完成简单分词"
    return result

#去掉值出现过一次的词
def delete1time(data_cut):
    frequency = defaultdict(int)
    for text in data_cut:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1] for text in data_cut]
    return texts

def cal_time(time):
    if time<60:
        return str(time) + 'secs '
    elif time<60*60:
        return str(time/(60.0))+' mins '
    else:
        return str(time/(60*60.0))+' hours '


def similiar(corpus,testDataSplit):
    # 使用 TF-IDF 模型对语料库建模
    tfidf = models.TfidfModel(corpus)
    m = len(testDataSplit)
    for i in range(2):
        print('测试第%d条数据' % i)
        testVec = dictionary.doc2bow(testDataSplit[i])
        index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
        sim = index[tfidf[testVec]]
        simNumList = sorted(enumerate(sim), key=lambda item: -item[1])

        sim_id=[]
        sim_doc=[]
        #f = open('/save_test.txt', 'a')
        for item in simNumList[0:22]:
            sim_id.append(id[item[0]])
            sim_doc.append(docs[item[0]][0].encode('utf-8'))
            #f.writelines(docs[item[1]][0])
        #f.close()
            print id[item[1]][0],docs[item[1]][0]
    return sim_id,sim_doc,simNumList



def out_txt(test_id,sim_id,sim_doc,path):
    result=[]
    for i,sim in enumerate(sim_id):
        test=test_id[i]
       # target=sim_doc[i]
        for j in range(2,22):
            result.append([test,sim[j]])
    df=DataFrame(result,columns=['source_id','target_id'])
    df.to_csv(path,sep='\t',index=False)

if __name__ == "__main__":

    start = time.clock()
    news_length = 100   #485687
    trainDataId,docs=extract_data('./train_data.csv',news_length)
    #data_cut=cut(docs)
    print "训练集加载完成！"
    map_id={}
    for i,idd in enumerate(trainDataId):
        map_id[idd[0]]=i
    testDataId,test_docs=extract_data('./test_data.csv',50)#50)
    #test_cut=cut(test_docs)
    print"测试集加载完成！"

    # 对训练集与测试集分词

    data_cut= cut(docs)
    trainDataSplit =delete1time(data_cut)
    print "已去除词频为1 的词"
    print('训练集分词完成')
    testDataSplit = cut(test_docs)
    print('测试集分词完成')

    # 制作语料库
    dictionary = corpora.Dictionary(trainDataSplit)  # 获取词袋
    corpus = [dictionary.doc2bow(doc) for doc in trainDataSplit]  # 制作语料库
    print('语料库制作完成')

    sim_id, sim_doc, simNumList = similiar(corpus, testDataSplit)
    path = './sx_result_57.txt'
    out_txt(test_id, sim_id, sim_doc, path)

    elapsed = (time.clock() - start)
    print('Time use',  cal_time(elapsed))
