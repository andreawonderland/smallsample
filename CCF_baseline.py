import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import BertTokenizer,AutoModel,AdamW,AutoConfig
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
from tqdm import tqdm
import copy
import torch.nn as nn
import os

# 如果有多卡可以指定使用哪张卡进行训练
os.environ["CUDA_VISIBLE_DEVICES"]='0'

# 模型训练参数设置
CONFIG = {
    'fold' : 10,
    'model_path': "../input/bert-base-chinese",
    'data_path' : "../input/train.json",
    'max_length' : 512,
    'train_batchsize' : 4,
    'vali_batchsize' : 4,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "learning_rate": 1e-5,
    "min_lr": 1e-6,
    "weight_decay": 1e-6,
    "T_max": 500,
    "seed":42,
    "num_class":36,
    "epoch_times":100,
    } 

# 设置随机种子
def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(CONFIG['seed'])


# 读取训练文件，处理输入
with open(CONFIG['data_path'], "r",encoding='UTF-8') as f:
    file_data = f.readlines()

df = pd.DataFrame(columns=['id','title','assignee','abstract','label_id'])
for each_json in file_data:
    json_dict = eval(each_json)
    df = df.append(json_dict,ignore_index = True)


df["label_id"] = df["label_id"].astype(int)
    
# 根据KFOLD划分数据
gkf = StratifiedKFold(n_splits=CONFIG['fold'])


for fold, (_, val_) in enumerate(gkf.split(X=df, y=df.label_id)):
    df.loc[val_, "kfold"] = int(fold)

df["kfold"] = df["kfold"].astype(int)
df.groupby('kfold')['label_id'].value_counts()
'''
三、模型相关的类创建
原始的数据包含ID,标题，公司名称，摘要等信息。在训练时，我们把这些信息连成一句话，输入模型进行训练：
text =  "这份专利的标题为：《{}》，由“{}”公司申请，详细说明如下：{}".format(title, assignee, abstract)
整个模型分类的思路为 BERT -> [cls]向量 -> dropout层 -> 全连接层 -> softmax 36种分类
'''
class CCFDataSet(Dataset):
    def __init__(self,df,tokenizer,max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.id = df["id"].values
        self.title = df["title"].values
        self.assignee = df["assignee"].values
        self.abstract = df["abstract"].values
        self.label_id = df["label_id"].values 

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        data_id = self.id[index]
        title = self.title[index]
        assignee = self.assignee[index]
        abstract = self.abstract[index]
        label = self.label_id[index]

        text =  "这份专利的标题为：《{}》，由“{}”公司申请，详细说明如下：{}".format(title, assignee, abstract)
        inputs = self.tokenizer.encode_plus(text, truncation = True, add_special_tokens = True, max_length = self.max_length)
        
        return { 'input_ids' : inputs['input_ids'],
                'attention_mask' : inputs['attention_mask'],
                'label' : label}

class CCFModel(nn.Module):
    
    def __init__(self):
        super(CCFModel,self).__init__()
        self.model = AutoModel.from_pretrained(CONFIG['model_path'])
        self.config = AutoConfig.from_pretrained(CONFIG['model_path'])
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
        # 获取[CLS]
        out = out[0][:, 0, :]
        out = self.dropout(out)
        outputs = self.fc(out)
        return outputs
'''
四、计算loss、labels、Collate相关类与函数
get_labels函数将模型output的结果通过softmax，整理为（预测标签，真实标签），便于后续计算F1分数。
'''
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
        
        return output
    
def criterion(outputs,labels):
    return nn.CrossEntropyLoss()(outputs,labels)

def get_labels(outputs,labels):
    # pred_labels 和 true_labels 便于后续计算F1分数
    outputs = F.softmax(outputs,dim=1).cpu().numpy()
    pred_labels = outputs.argmax(1)
    pred_labels = pred_labels.tolist()
    
    true_labels = labels.cpu().tolist()
    return pred_labels,true_labels
'''
五、训练和验证函数
训练和验证的过程比较类似，唯一的区别就是验证的时候额外计算了score（F1分数）。后续每个epoch结束都会判断score分数，如果当前分数比best_score有提升则进行保存。
'''
def train_one_epoch(model,train_loader,optimizer,scheduler,epoch,device):
    model.train()
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, data in bar:
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        labels = data['label'].to(device, dtype=torch.long)
        
        batch_size = ids.size(0)
        outputs = model(ids,mask)
        
        loss = criterion(outputs,labels)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        if scheduler is not None:
            scheduler.step()
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss/dataset_size
        
        bar.set_postfix(Epoch=epoch,Train_loss=epoch_loss,LR=optimizer.param_groups[0]['lr'])
    return epoch_loss

@torch.no_grad()
def valid_one_epoch(model,vali_loader,optimizer,epoch,device):
    model.eval()
    dataset_size = 0
    running_loss = 0.0
    
    pred_labels =[]
    true_labels =[]
    
    bar = tqdm(enumerate(vali_loader), total=len(vali_loader))
    for step, data in bar:
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        labels = data['label'].to(device, dtype=torch.long)
        
        batch_size = ids.size(0)
        outputs = model(ids,mask)
        
        loss = criterion(outputs,labels)
        
        batch_pred_labels, true_pred_labels = get_labels(outputs,labels)
        pred_labels += batch_pred_labels
        true_labels += true_pred_labels
        epoch_score = f1_score(pred_labels,true_labels,average='macro')
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss/dataset_size
        
        bar.set_postfix(Epoch=epoch,Valid_loss=epoch_loss,F_score=epoch_score,LR=optimizer.param_groups[0]['lr'])
    return epoch_loss, epoch_score
'''
六、一切都准备完毕，开始训练啦
获取tokenizer,model,scheduler,optimizer; 划分train_loader,vali_loader等等操作。
'''
tokenizer = BertTokenizer.from_pretrained(CONFIG['model_path'])
collate_fn = Collate(tokenizer,True)     

# 拆分训练集和验证集,默认第一个fold作为验证集，后九个为训练集，则训练集占90%，验证集占10%
fold = 0
train_data = df[df["kfold"] != fold].reset_index(drop=True)
vali_data = df[df["kfold"] == fold].reset_index(drop=True)

train_dataset = CCFDataSet(train_data,tokenizer,CONFIG['max_length'])
vali_dataset = CCFDataSet(vali_data,tokenizer,CONFIG['max_length'])

train_loader = DataLoader(train_dataset,batch_size=CONFIG['train_batchsize'],collate_fn=collate_fn,
                          shuffle = True, drop_last = True, pin_memory = False, num_workers=8)
vali_loader = DataLoader(vali_dataset,batch_size=CONFIG['vali_batchsize'],collate_fn=collate_fn,
                          shuffle = False, pin_memory = False, num_workers=8)


model = CCFModel()
model.to(CONFIG['device'])

optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['T_max'], eta_min=CONFIG['min_lr'])

# 根据之前设置好的epoch_times，进行训练并保存模型。模型保存的结果为best_weights.bin

# 数据和参数准备结束，开始训练
if torch.cuda.is_available():
    print("GPU: {}\n".format(torch.cuda.get_device_name()))
    
best_weights = copy.deepcopy(model.state_dict())
best_score = 0

for epoch in range(CONFIG['epoch_times']):
    train_loss = train_one_epoch(model,train_loader,optimizer,scheduler,epoch,CONFIG['device'])
    valid_loss, valid_score = valid_one_epoch(model,vali_loader,optimizer,epoch,CONFIG['device'])
    
    
    if valid_score >= best_score:
        print(f"Validation Score Improved ({best_score} ---> {valid_score})")
        best_score = valid_score
        best_weights = copy.deepcopy(model.state_dict())
        
        PATH = f"best_weights.bin"
        torch.save(model.state_dict(),PATH)
        
print("Best F1 score:" + str(best_score))         



