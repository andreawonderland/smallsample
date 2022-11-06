import pandas as pd
from sklearn.model_selection import StratifiedKFold

with open('G:/西电研究生/smallsample_contest/train.json', "r",encoding='UTF-8') as f:
    file_data = f.readlines()
df = pd.DataFrame(columns=['id','title','assignee','abstract','label_id'])
for each_json in file_data:
    json_dict = eval(each_json)
    df = df.append(json_dict,ignore_index = True)
df["label_id"] = df["label_id"].astype(int)
# 根据KFOLD划分数据
gkf = StratifiedKFold(n_splits=10)
for fold, (_, val_) in enumerate(gkf.split(X=df, y=df.label_id)):
    df.loc[val_, "kfold"] = int(fold)
df["kfold"] = df["kfold"].astype(int)
df.groupby('kfold')['label_id'].value_counts()
print('1')