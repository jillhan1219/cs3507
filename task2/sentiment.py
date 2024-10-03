import argparse
import logging
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Tokenizer, BertTokenizer, AlbertTokenizer, GPT2Config
import os
from GPT2Model import GPT2ModelWithLearnablePositionalEncoding,CustomGPT2Model,MAMBADecoder

class GPT2ForSentiment(torch.nn.Module):
    def __init__(self, config, num_labels, args):
        super(GPT2ForSentiment, self).__init__()
        # embedding
        #if args.embedding == 'spe':
            #self.gpt2 = GPT2Model.from_pretrained("./models")
        #elif args.embedding == 'lpe':
            #self.gpt2 = GPT2ModelWithLearnablePositionalEncoding(config)
        #else:
            #raise ValueError("Embdding: spe or lpe")

        # attention
        #if args.attention == 'self':
            #self.gpt2 = GPT2Model.from_pretrained("./models")
        #elif args.attention == 'linear1' or args.attention == 'linear2':
            #self.gpt2 = CustomGPT2Model(config, args.attention)
        #elif args.attention == 'nystrom':
            #self.gpt2 = CustomGPT2Model(config, "nystrom")
        #else:
            #raise ValueError("Attention: self, linear or nystrom")
        
        # mamba
        self.mamba = False
        if args.mamba:
            self.gpt2 = MAMBADecoder(config)
            self.mamba = True

        self.lm_head = torch.nn.Linear(config.n_embd, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, labels=None):
        if not self.mamba:
            hidden_state = self.gpt2(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,-1]
        else:
            hidden_state = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)[:,-1]
        
        return self.lm_head(hidden_state)


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, max_len=256):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

def evaluate_model(model, test_dataset, batch_size=8):
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for text, label in tqdm(test_dataloader, desc="Test:"):
            input_ids = text['input_ids'].squeeze().cuda()
            attention_mask = text['attention_mask'].squeeze().cuda()
            labels = label.cuda()
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs, 1)
            #predicted = predicted.argmax(dim=1)
            #print("predicted:", predicted)
            total_samples += labels.size(0)
            true_labels = torch.argmax(labels,1)
            total_correct += (predicted == true_labels).sum().item()
    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy}")
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Sentiment Classification with Different Tokenizers')
    parser.add_argument('--tokenizer', type=str, choices=['gpt2', 'bert', 'albert'], default='gpt2', help='Choose tokenizer: gpt2, bert, albert')
    parser.add_argument('--bz', type=int, default=8, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--log_path', type=str, default='logs', help='log name')
    parser.add_argument('--log_name', type=str, default='training.log', help='log name')
    parser.add_argument('--save', type=bool, default=False, help='save pth or not')
    parser.add_argument('--embedding', type=str, default='spe', help='position embedding')
    parser.add_argument('--attention', type=str, default='self', help='attention type')
    parser.add_argument('--mamba', type=bool, default=False, help='mamba or not')
    args = parser.parse_args()

    if args.tokenizer == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained("./models")
        tokenizer.pad_token = tokenizer.eos_token
    elif args.tokenizer == 'bert':
        tokenizer = BertTokenizer.from_pretrained('./models/tokenizer/word')#bert-base-uncased
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    elif args.tokenizer == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained('./models/tokenizer/sentence')#albert-base-v2
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    max_length = 256

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    logging.basicConfig(
        filename= os.path.join(args.log_path, args.log_name),
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger()
    logger.info(args)

    train_path = './dataset/train.csv'
    test_path = './dataset/test.csv'
    train_df = pd.read_csv(train_path, encoding='iso-8859-1').dropna()
    test_df = pd.read_csv(test_path, encoding='iso-8859-1').dropna()

    mapping = {'positive': torch.tensor([1,0,0]), 'negative': torch.tensor([0,1,0]), 'neutral': torch.tensor([0,0,1])}
    train_labels = train_df['sentiment'].map(mapping).to_numpy()
    test_labels = test_df['sentiment'].map(mapping).to_numpy()

    def process_text(ex):
        return tokenizer(ex, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')

    train_texts = train_df['text'].dropna().map(process_text).to_numpy()
    test_texts = test_df['text'].dropna().map(process_text).to_numpy()

    train_dataset = SentimentDataset(train_texts, train_labels)
    test_dataset = SentimentDataset(test_texts, test_labels)
    dataloader = DataLoader(train_dataset, batch_size=args.bz, shuffle=True)

    #model = GPT2ForSentiment(gpt2.config, 3).cuda()
    config = GPT2Config.from_pretrained('./models')
    model = GPT2ForSentiment(config, 3, args).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = CrossEntropyLoss()

    best_acc = 0
    all_loss = []
    for epoch in tqdm(range(args.epochs+1)):
        t_data = dataloader
        #accuracy = evaluate_model(model, test_dataset)
        for text, label in tqdm(t_data, desc = "1 epoch:"):
            optimizer.zero_grad()
            attention_mask = text['attention_mask'].squeeze()
            #print(f'attention mask:{attention_mask}')
            #print(f'attention mask shape:{attention_mask.shape}')
            output = model(text['input_ids'].squeeze().cuda(), attention_mask=attention_mask.cuda())
            labels = label.cuda()
            loss = loss_fn(output, labels.argmax(dim = 1))
            loss.backward()
            optimizer.step()
            all_loss.append(loss.item())
        print(f'Epoch {epoch}, Loss: {np.mean(all_loss)}')
        logger.info(f'Epoch {epoch}, Loss: {np.mean(all_loss)}')

        accuracy = evaluate_model(model, test_dataset)
        if accuracy > best_acc:
            if args.save:
                torch.save(model.state_dict(), 'sentiment.pt')
                torch.save(model, 'sentiment.pth')
            best_acc = accuracy
            logger.info(f'Epoch {epoch}, acc: {best_acc}')

if __name__ == "__main__":
    main()
