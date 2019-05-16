# -*- coding: utf-8 -*-

# @Time    : 2019-05-06 14:08
# @Author  : jian
# @File    : two.py

import jieba.analyse as analyse
import pandas as pd

"""
基于 TF-IDF 算法的关键词抽取
import jieba.analyse
•jieba.analyse.extract_tags(sentence, topK=20, withWeight=False, allowPOS=())◾sentence 为待提取的文本
◾topK 为返回几个 TF/IDF 权重最大的关键词，默认值为 20
◾withWeight 为是否一并返回关键词权重值，默认值为 False
◾allowPOS 仅包括指定词性的词，默认值为空，即不筛选
"""

# df = pd.read_csv("data/technology_news.csv", encoding='utf-8')
# df = df.dropna()
# lines = df.content.values.tolist()
# content = "".join(lines)
# print("  ".join(analyse.extract_tags(content, topK=10, withWeight=False, allowPOS=())))


# df = pd.read_csv("data/military_news.csv", encoding='utf-8')
# df = df.dropna()
# lines = df.content.values.tolist()
# content = "".join(lines)
# print("  ".join(analyse.extract_tags(content, topK=3, withWeight=False, allowPOS=())))

"""
基于 TextRank 算法的关键词抽取
•jieba.analyse.textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v')) 
直接使用，接口相同，注意默认过滤词性。
•jieba.analyse.TextRank() 新建自定义 TextRank 实例
算法论文： TextRank: Bringing Order into Texts

基本思想:
•将待抽取关键词的文本进行分词
•以固定窗口大小(默认为5，通过span属性调整)，词之间的共现关系，构建图
•计算图中节点的PageRank，注意是无向带权图

"""

# df = pd.read_csv("./data/military_news.csv", encoding='utf-8')
# df = df.dropna()
# lines = df.content.values.tolist()
# content = "".join(lines)
#
# print("  ".join(analyse.textrank(content, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))))
# print("---------------------我是分割线----------------")
# print("  ".join(analyse.textrank(content, topK=20, withWeight=False, allowPOS=('ns', 'n'))))

"""
LDA主题模型
咱们来用LDA主题模型建模，看看这些新闻主要在说哪些topic。
首先我们要把文本内容处理成固定的格式，一个包含句子的list，list中每个元素是分词后的词list。类似下面这个样子。
[[第，一，条，新闻，在，这里],[第，二，条，新闻，在，这里],[这，是，在，做， 什么],...]
"""

from gensim import corpora, models, similarities
import gensim

stopwords = pd.read_csv("data/stopwords.txt", index_col=False, quoting=3, sep="\t", names=['stopword'],
                        encoding='utf-8')
stopwords = stopwords['stopword'].values

import jieba
import pandas as pd

df = pd.read_csv("./data/technology_news.csv", encoding='utf-8')
df = df.dropna()
lines = df.content.values.tolist()

sentences = []
for line in lines:
    try:
        segs = jieba.lcut(line)
        segs = filter(lambda x: len(x) > 1, segs)
        segs = filter(lambda x: x not in stopwords, segs)
        sentences.append(segs)
    except Exception as e:
        print(line)
        continue

for word in sentences[5]:
    print(word)

# 词袋模型
dictionary = corpora.Dictionary(sentences)
corpus = [dictionary.doc2bow(sentence) for sentence in sentences]

print(corpus[5])

# LDA建模
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)

# 我们查一下第3号分类，其中最常出现的单词是：
print(lda.print_topic(3, topn=5))

for topic in lda.print_topics(num_topics=20, num_words=8):
    print(topic[1])


# lda.get_document_topics(bow)