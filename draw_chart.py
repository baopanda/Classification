import json
import os
from collections import Counter
from os.path import join

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


pd.set_option('display.max_columns', 7)

with open("json_train_new.json","r",encoding="utf-8") as f:
    d = json.load(f)

data = []
label = []
count = 0
for i in d:
    # data.append(ViTokenizer.tokenize(i['content']))
    data.append(i['content'])
    label.append(i['label'])
    # count=+1
    # print(count)

print("#Data: ",len(data))
print("#Label: ",len(label))
df = pd.DataFrame({"datas": data, "label": label}).sort_values("label")
label = Counter(df.label)
count = []
for i in label:
    count.append(label[i])
print(count)

g = sns.countplot(x="label", data=df)
plt.tight_layout()
# plt.annotate(label,count)

# plt.savefig(join("images", "image.png"))
plt.show()