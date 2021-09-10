import json, os
from sys import base_exec_prefix
from torch.utils.data import DataLoader
from data_utils import TripletDataset, collate_fn
from transformers import AutoModel, AutoTokenizer


bert_base_model = AutoModel.from_pretrained("distilroberta-base")
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")


DATA_PATH = '../data/final'
MAX_LEN = 512
BATCH_SIZE = 32

def load_data(data_path, data_type):
    with open(os.path.join(data_path, data_type), 'r', encoding='utf-8') as f:
        return json.load(f)


train_dataset = TripletDataset(load_data(DATA_PATH, 'train.json'), tokenizer, max_len=MAX_LEN)
test_dataset = TripletDataset(load_data(DATA_PATH, 'test.json'), tokenizer, max_len=MAX_LEN)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn)

for batch in train_dataloader:
    anchor_ids, anchor_mask, positive_ids, positive_mask, negative_ids, negative_mask = batch
    print(batch)

print(1)