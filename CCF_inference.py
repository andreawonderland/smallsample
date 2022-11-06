import json
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.metrics import f1_score
from transformers import BertTokenizer,AutoModel,AdamW,AutoConfig,AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
from tqdm import tqdm
import time
import copy
import numpy as np
from collections import defaultdict
import torch.nn as nn
import os


os.environ["CUDA_VISIBLE_DEVICES"]='1'

CONFIG = {
    'weight_paths': "best_weights.bin",
    'model_path' : "../input/bert-base-chinese",
    'data_path' : "../input/testA.json",
    'max_length' : 512,
    'test_batchsize' : 4,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "seed":42,
    "num_class":36,
    } 

with open(CONFIG['data_path'], "r",encoding='UTF-8') as f:
    file_data = f.readlines()

df = pd.DataFrame(columns=['id','title','assignee','abstract'])
for each_json in file_data:
    json_dict = eval(each_json)
    df = df.append(json_dict,ignore_index = True)


class CCFDataSet(Dataset):
    def __init__(self,df,tokenizer,max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.id = df["id"].values
        self.title = df["title"].values
        self.assignee = df["assignee"].values
        self.abstract = df["abstract"].values

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        data_id = self.id[index]
        title = self.title[index]
        assignee = self.assignee[index]
        abstract = self.abstract[index]
        
        text =  "这份专利的标题为：《{}》，由“{}”公司申请，详细说明如下：{}".format(title, assignee, abstract)
        
        inputs = self.tokenizer.encode_plus(text, truncation = True, add_special_tokens = True, max_length = self.max_length)
        
        return { 'input_ids' : inputs['input_ids'],
                'attention_mask' : inputs['attention_mask'],
                'data_id': data_id}

class Collate():
    def __init__(self,tokenizer,isTrain=True):
        self.tokenizer = tokenizer
        self.isTrain = isTrain
    def __call__(self,batch):
        output = dict()
        output['input_ids'] = [sample['input_ids'] for sample in batch]
        output['attention_mask'] = [sample['attention_mask'] for sample in batch]
        
        if self.isTrain:
            output['label'] = [sample['label'] for sample in batch]
        else:
            output['data_id'] = [sample['data_id'] for sample in batch]
        
        btmax_len = max([len(i) for i in output['input_ids']])
        
        # 手动进行pad填充
        if self.tokenizer.padding_side == 'right':
            output['input_ids'] = [ i + [self.tokenizer.pad_token_id] * (btmax_len - len(i)) for i in output['input_ids']]
            output['attention_mask'] = [ i + [0] * (btmax_len - len(i)) for i in output['attention_mask']]
        else:
            output['input_ids'] = [ [self.tokenizer.pad_token_id] * (btmax_len - len(i)) + i for i in output['input_ids']]
            output['attention_mask'] = [ [0] * (btmax_len - len(i)) + i for i in output['attention_mask']]

        output['input_ids'] = torch.tensor(output['input_ids'],dtype = torch.long)
        output['attention_mask'] = torch.tensor(output['attention_mask'],dtype = torch.long)
        
        if self.isTrain:
            output['label'] = torch.tensor(output['label'],dtype = torch.long)
        else:
            output['data_id'] = output['data_id']
        
        return output
    

class CCFModel(nn.Module):
    
    def __init__(self, model_path):
        super(CCFModel,self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.config = AutoConfig.from_pretrained(model_path)
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(self.config.hidden_size,CONFIG['num_class'])
        self._init_weights(self.fc)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm): 
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=False)
        out = out[0][:, 0, :]
        out = self.dropout(out)
        outputs = self.fc(out)
        return outputs

def get_score(outputs):
    # pred_labels 和 true_labels 便于后续计算F1分数
    outputs = F.softmax(outputs,dim=1).cpu().numpy()
    pred_labels = outputs.argmax(1)
    pred_labels = pred_labels.tolist()
    
    return pred_labels
# 七、使用模型预测结果
# 首先在本地读取保存的最优bin文件，然后再进行inference，使用到了很多上面训练时用到的函数，这里只把主要过程展示出来，完整的的inference文件已经放入附件中。
@torch.no_grad()
def test_fn(model,test_loader,device):
    model.eval()
    
    pred_labels =[]
    id_list = []
    bar = tqdm(enumerate(test_loader), total=len(test_loader))
    for step, data in bar:
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        
        data_id = data['data_id']
        id_list += data_id
        
        outputs = model(ids,mask)
        batch_pred_labels = get_score(outputs)
        pred_labels += batch_pred_labels
        
    return pred_labels,id_list


tokenizer = BertTokenizer.from_pretrained(CONFIG['model_path'])
collate_fn = Collate(tokenizer,False)

test_dataset = CCFDataSet(df,tokenizer,CONFIG['max_length'])
test_loader = DataLoader(test_dataset,batch_size=CONFIG['test_batchsize'],collate_fn=collate_fn,
                              shuffle = False, pin_memory = False, num_workers=8)
models_results_dict = {}

model = CCFModel(CONFIG['model_path'])
model.to(CONFIG['device'])
model.load_state_dict(torch.load(CONFIG['weight_paths']))

preds,id_list = test_fn(model,test_loader,CONFIG['device'])
models_results_dict["id"] = id_list
models_results_dict["label"] = preds
    
    
df_result = pd.DataFrame(models_results_dict)
df_result.to_csv('result.csv',index=False)
