import json

from pyvi import ViTokenizer
from nltk.corpus import stopwords

with open("json_train_new.json","r",encoding="utf-8") as f:
    d = json.load(f)

data = []
label = []
# count = 0
for i in d:
    # data.append(ViTokenizer.tokenize(i['content']))
    data.append(i['content'])
    label.append(i['label'])


SPECIAL_CHARACTER = '%@$=+-!;/()*"&^:#|\n\t\''
with open("datas_stopword.txt",'w',encoding='utf-8') as file:
    for i in data:
        i = i.strip(SPECIAL_CHARACTER)
        i = ViTokenizer.tokenize(i)
        # print(i)
        for word in i:
            if word in stopwords.words('vietnam') and len(word)>20:
                i.strip(word)
        print(i)
        file.write(i+"\n")
with open("labels_new.txt",'w',encoding='utf-8') as file:
    for i in label:
        print(i)
        file.write(str(i)+"\n")

