import json, os
from sys import base_exec_prefix
from numpy.core.numeric import Inf
from torch.utils.data import DataLoader
from data_utils import TripletDataset, collate_fn
from transformers import AutoModel, AutoTokenizer, AdamW

from torch.nn import TripletMarginLoss
import torch

from tqdm import tqdm


DATA_PATH = '../data/final'
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHES = 10
MODE = "train"

def load_data(data_path, data_type):
    with open(os.path.join(data_path, data_type), 'r', encoding='utf-8') as f:
        return json.load(f)




def trainer(EPOCHES, model):
    
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    learning_rate = 1e-5
    optimizer = AdamW(model.parameters(), learning_rate)
    criterion = TripletMarginLoss()

    train_dataset = TripletDataset(load_data(DATA_PATH, 'train.json'), tokenizer, max_len=MAX_LEN)
    test_dataset = TripletDataset(load_data(DATA_PATH, 'test.json'), tokenizer, max_len=MAX_LEN)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    train(EPOCHES, model, train_dataloader, test_dataloader, optimizer, criterion)


def train(epoches, model, train_dataloader, test_dataloader, optimizer, criterion):
    best_valid_loss = float('inf')
    for epoch in range(1, epoches+1):
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            if step % 10 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.  Train Loss: {:.2f}'.format(step, len(train_dataloader), total_loss/step))

            anchor_ids, anchor_mask, positive_ids, positive_mask, negative_ids, negative_mask = [_.cuda() for _ in batch]
            anchor_output = model(anchor_ids, anchor_mask).last_hidden_state[:, 0, :]
            positive_output = model(positive_ids, positive_mask).last_hidden_state[:, 0, :]
            negative_output = model(negative_ids, negative_mask).last_hidden_state[:, 0, :]

            loss = criterion(anchor_output, positive_output, negative_output)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        torch.cuda.empty_cache()
        # avg_loss = total_loss / step

        model.eval()
        valid_total_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(test_dataloader):
                anchor_ids, anchor_mask, positive_ids, positive_mask, negative_ids, negative_mask = [_.cuda() for _ in batch]
                anchor_output = model(anchor_ids, anchor_mask).last_hidden_state[:, 0, :]
                positive_output = model(positive_ids, positive_mask).last_hidden_state[:, 0, :]
                negative_output = model(negative_ids, negative_mask).last_hidden_state[:, 0, :]
    
                loss = criterion(anchor_output, positive_output, negative_output)
                valid_total_loss += loss
        
        avg_valid_loss = valid_total_loss / step
        # if avg_valid_loss < best_valid_loss:
        #     best_valid_loss = avg_valid_loss
        print("  >> Saving model... epoch: {}".format(epoch))
        torch.save(model.state_dict(), '../result/Triplet_bert_val_loss_{}.pt'.format(epoch))

        print("  >> Validation Loss: {:.5f}".format(avg_valid_loss))


def main():

    bert_base_model = AutoModel.from_pretrained("monologg/koelectra-base-v3-discriminator").cuda()
    
    if MODE == 'train':
        trainer(EPOCHES, bert_base_model)


if __name__ == "__main__":
    main()
