import json

from pyvi import ViTokenizer
from nltk.corpus import stopwords

import StopWords

with open("json_train_new.json","r",encoding="utf-8") as f:
    d = json.load(f)

data = []
label = []

for i in d:
    data.append(i['content'])
    label.append(i['label'])

SPECIAL_CHARACTER = '%@$=+-!;/()*"&^:♥<>#|\n\t\''
with open("datas_stopword1.txt",'w',encoding='utf-8') as file:
    for i in data:
        my_words = i.split(" ")
        for word1 in i:
            if word1 in SPECIAL_CHARACTER:
                i = i.replace(word1, "")
                i = i.replace("  ", " ")
        for word in my_words:
            if len(word) > 20:
                i = i.replace(word, "")
                i = i.replace("  ", " ")
        i = ViTokenizer.tokenize(i)
        my_words = i.split(" ")
        for word in my_words:
            if word in StopWords.STOP_WORDS:
                i = i.replace(word, "")
                i = i.replace("  ", " ")
        i = i.lower()
        file.write(i+"\n")

with open("labels_new1.txt",'w',encoding='utf-8') as file:
    for i in label:
        file.write(str(i)+"\n")

