from torch.utils.data import Dataset
import torch

class TripletDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_pair = self.dataset[idx]
        
        anchor = self.tokenizer(data_pair[0], add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False, padding='max_length', return_attention_mask=True, return_tensors='pt', truncation=True)
        positive = self.tokenizer(data_pair[1], add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False, padding='max_length', return_attention_mask=True, return_tensors='pt', truncation=True)
        negative = self.tokenizer(data_pair[2], add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False, padding='max_length', return_attention_mask=True, return_tensors='pt', truncation=True)

        anchor_ids, anchor_mask = anchor['input_ids'], anchor['attention_mask']
        positive_ids, positive_mask = positive['input_ids'], positive['attention_mask']
        negative_ids, negative_mask = negative['input_ids'], negative['attention_mask']

        return (anchor_ids, anchor_mask, positive_ids, positive_mask, negative_ids, negative_mask)

def collate_fn(batch):

    def merge(input_ids, attention_mask):
        lengths = [_.size(1) for _ in input_ids]
        padded_ids = torch.zeros(len(input_ids), max(lengths)).long()
        padded_mask = torch.zeros(len(input_ids), max(lengths)).long()
        for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
            end = lengths[i]
            padded_ids[i, :end] = ids[0, :end]
            padded_mask[i, :end] = mask[0, :end]
        return padded_ids, padded_mask
    
    anchor_ids, anchor_mask = [], []
    positive_ids, positive_mask = [], []
    negative_ids, negative_mask = [], []

    for data in batch:
        anchor_ids.append(data[0])
        anchor_mask.append(data[1])
        positive_ids.append(data[2])
        positive_mask.append(data[3])
        negative_ids.append(data[4])
        negative_mask.append(data[5])
    
    anchor_ids, anchor_mask = merge(anchor_ids, anchor_mask)
    positive_ids, positive_mask = merge(positive_ids, positive_mask)
    negative_ids, negative_mask = merge(negative_ids, negative_mask)
    
    return anchor_ids, anchor_mask, positive_ids, positive_mask, negative_ids, negative_mask