# -*- coding: utf-8 -*-

# @Time    : 2019-05-15 9:44
# @Author  : jian
# @File    : base_nltk.py
import nltk
from nltk.corpus import brown  # 文章的集合

# nltk.download('brown')

# print(brown.categories())
# print(len(brown.sents()))
# print(len(brown.words()))

sentence = "hello, world"
token = nltk.word_tokenize(sentence)
print(token)
