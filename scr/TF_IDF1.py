# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 21:49:31 2018
@author: suxue
"""
import jieba
from gensim import corpora, models, similarities
import pandas as pd
import re
import time

# 加载数据集
'''
加载训练集方式：
    loadDataSet('train_data.csv', 'UTF-8')
加载测试集方式：
    loadDataSet('test_data.csv', 'gbk')

'''


def loadDataSet(filename, encode):
    dataSet = pd.read_csv(filename, encoding=encode)
    m, n = dataSet.shape
    data = dataSet.values[:, -1]
    dataId = dataSet.values[:, 0]
    return data.reshape((m, 1)), dataId.reshape((m, 1))


# numpy 数组转化为 list
def ndarrayToList(dataArr):
    dataList = []
    m, n = dataArr.shape
    for i in range(m):
        for j in range(n):
            dataList.append(dataArr[i, j])
    return dataList


# 去掉字符串特殊符号
def removeStr(listData):
    strData = "".join(listData)
    removeStrData = re.sub("[\s+\!\/_,$^*()+\"\']+:|[+——！，,《》“”〔【】；：。？、~@#￥……&*（）]+", "", strData)
    return removeStrData


# 对数据集分词
def wordSplit(data):
    word = ndarrayToList(data)
    m = len(word)
    wordList = []
    for i in range(m):
        # rowListRemoveStr = removeStr(word[i])
        rowListRemoveStr = word[i]
        rowList = [eachWord for eachWord in jieba.cut(rowListRemoveStr)]
        wordList.append(rowList)
    return wordList


if __name__ == "__main__":

    start = time.clock()

    with open("result.txt", "a") as fr:
        fr.write('source_id')
        fr.write('\t')
        fr.write('target_id')
        fr.close

    # 加载训练集与测试集
    trainData, trainDataId = loadDataSet('train_data.csv', 'UTF-8')
    testData, testDataId = loadDataSet('test_data.csv', 'gbk')
    print('训练集、测试集加载完成')

    # 对训练集与测试集分词
    trainDataSplit = wordSplit(trainData)
    print('训练集分词完成')
    testDataSplit = wordSplit(testData)
    print('测试集分词完成')

    # 制作语料库
    dictionary = corpora.Dictionary(trainDataSplit)  # 获取词袋
    corpus = [dictionary.doc2bow(doc) for doc in trainDataSplit]  # 制作语料库
    print('语料库制作完成')

    # 使用 TF-IDF 模型对语料库建模
    tfidf = models.TfidfModel(corpus)
    m = len(testDataSplit)

    for i in range(m):
        print('测试第%d条数据' % i)
        testVec = dictionary.doc2bow(testDataSplit[i])
        index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
        sim = index[tfidf[testVec]]
        simNumList = sorted(enumerate(sim), key=lambda item: -item[1])

        with open("result.txt", "a") as fr:
            for j in range(21):
                if str(int(testDataId[i])) == str(int(simNumList[j][0] + 1)):
                    continue
                fr.write('\n')
                fr.write(str(int(testDataId[i])))
                fr.write('\t')
                fr.write(str(int(simNumList[j][0] + 1)))

    elapsed = (time.clock() - start)
    print('Time use', elapsed)