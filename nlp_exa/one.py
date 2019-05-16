# -*- coding: utf-8 -*-

# @Time    : 2019-05-06 13:56
# @Author  : jian
# @File    : one.py

# coding:utf-8
__author__ = 'Hanxiaoyang'

import warnings

warnings.filterwarnings("ignore")
import jieba  # 分词包
import numpy  # numpy计算包
import codecs  # codecs提供的open方法来指定打开的文件的语言编码，它会在读取的时候自动转换为内部unicode
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
from wordcloud import WordCloud  # 词云包

# 导入娱乐新闻数据 分词
df = pd.read_csv("data/entertainment_news.csv", encoding='utf-8')
df = df.dropna()
content = df.content.values.tolist()
# jieba.load_userdict(u"data/user_dic.txt")
segment = []
for line in content:
    try:
        segs = jieba.lcut(line)
        for seg in segs:
            if len(seg) > 1 and seg != '\r\n':
                segment.append(seg)
    except:
        print(line)
        continue

words_df = pd.DataFrame({'segment': segment})
# words_df.head()
stopwords = pd.read_csv("data/stopwords.txt", index_col=False, quoting=3, sep="\t", names=['stopword'],
                        encoding='utf-8')  # quoting=3全不引用
# stopwords.head()
words_df = words_df[~words_df.segment.isin(stopwords.stopword)]

words_stat = words_df.groupby(by=['segment'])['segment'].agg({"计数": numpy.size})
words_stat = words_stat.reset_index().sort_values(by=["计数"], ascending=False)
print(words_stat.head())

wordcloud = WordCloud(font_path="data/simhei.ttf", background_color="white", max_font_size=80)
word_frequence = {x[0]: x[1] for x in words_stat.head(1000).values}
wordcloud = wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)
plt.show()

from scipy.misc import imread

matplotlib.rcParams['figure.figsize'] = (15.0, 15.0)
from wordcloud import WordCloud, ImageColorGenerator

bimg = imread('image/entertainment.jpeg')
wordcloud = WordCloud(background_color="white", mask=bimg, font_path='data/simhei.ttf', max_font_size=200)
word_frequence = {x[0]: x[1] for x in words_stat.head(1000).values}
wordcloud = wordcloud.fit_words(word_frequence)
bimgColors = ImageColorGenerator(bimg)
plt.axis("off")
plt.imshow(wordcloud.recolor(color_func=bimgColors))
plt.show()


df = pd.read_csv("data/sports_news.csv", encoding='utf-8')
df = df.dropna()
content=df.content.values.tolist()
#jieba.load_userdict(u"data/user_dic.txt")
segment=[]
for line in content:
    try:
        segs=jieba.lcut(line)
        for seg in segs:
            if len(seg)>1 and seg!='\r\n':
                segment.append(seg)
    except:
        print( line)
        continue






matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)
words_df=pd.DataFrame({'segment':segment})
#words_df.head()
stopwords=pd.read_csv("data/stopwords.txt",index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')#quoting=3全不引用
#stopwords.head()
words_df=words_df[~words_df.segment.isin(stopwords.stopword)]
words_stat=words_df.groupby(by=['segment'])['segment'].agg({"计数":numpy.size})
words_stat=words_stat.reset_index().sort_values(by=["计数"],ascending=False)
words_stat.head()
wordcloud=WordCloud(font_path="data/simhei.ttf",background_color="black",max_font_size=80)
word_frequence = {x[0]:x[1] for x in words_stat.head(1000).values}
wordcloud=wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)
plt.show()


from scipy.misc import imread
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
from wordcloud import WordCloud,ImageColorGenerator
bimg=imread('image/sports.jpeg')
wordcloud=WordCloud(background_color="white",mask=bimg,font_path='data/simhei.ttf',max_font_size=200)
word_frequence = {x[0]:x[1] for x in words_stat.head(1000).values}
wordcloud=wordcloud.fit_words(word_frequence)
bimgColors=ImageColorGenerator(bimg)
plt.axis("off")
plt.imshow(wordcloud.recolor(color_func=bimgColors))
plt.show()