# -*- coding: utf-8 -*-

# @Time    : 2019-05-15 8:19
# @Author  : jian
# @File    : base4.py
from gtts import gTTS
import os
tts = gTTS(text='您好，我是您的私人助手，我叫小辣椒', lang='zh-tw')
tts.save("hello.mp3")
os.system("mpg321 hello.mp3")