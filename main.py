import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
from sklearn.model_selection import train_test_split
import os
from rouge import Rouge
import math
import collections

import pandas as pd
import numpy as np
import csv

from Seq2seqLSTM import Seq2SeqLSTM, Decoder, Encoder
from Seq2seqBART import Seq2SeqBART


class CustomDataset():
    def __init__(self, descriptions, diagnoses):
        self.descriptions = descriptions
        self.diagnoses = diagnoses

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        description = self.descriptions.iloc[idx]['description']
        diagnosis = self.diagnoses.iloc[idx]
        return {'description': description, 'diagnosis': diagnosis}

class ProjectRun:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default='Seq2SeqBART', type=str)
        parser.add_argument('--lr', default=0.05, type=float)
        parser.add_argument('--dropout', default=0.0, type=float)
        parser.add_argument('--epoch', default=1, type=int)
        parser.add_argument('--batch_size', default=16, type=int)
        args = parser.parse_args()

        self.model = args.model
        self.lr = float(args.lr)
        self.dropout = float(args.dropout)
        self.epoch = int(args.epoch)
        self.batch_size = int(args.batch_size)

        self.train_iterator = None
        self.val_iterator = None
        self.test_iterator = None

        self.train_size = 0
        self.test_size = 0
        self.val_size = 0

    def dataset(self):
        # 加载数据
        train_data = pd.read_csv("./train.csv", sep=',', encoding='utf-8')
        train_data_description = train_data[['description']]
        train_data_diagnosis = train_data['diagnosis']
        test_data = pd.read_csv("./test.csv", sep=',', encoding='utf-8')
        test_data_description = test_data[['description']]
        test_data_diagnosis = test_data['diagnosis']

        train_data_description, val_data_description, train_data_diagnosis, val_data_diagnosis = train_test_split(train_data_description, train_data_diagnosis, test_size=0.2, random_state=42)

        train_dataset = CustomDataset(train_data_description, train_data_diagnosis)
        self.train_iterator = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)

        val_dataset = CustomDataset(val_data_description, val_data_diagnosis)
        self.val_iterator = DataLoader(val_dataset, shuffle=True, batch_size=self.batch_size)

        test_dataset = CustomDataset(test_data_description, test_data_diagnosis)
        self.test_iterator = DataLoader(test_dataset, shuffle=True, batch_size=self.batch_size)


    def train(self):
        torch.manual_seed(seed=12)

        INPUT_DIM = 512
        OUTPUT_DIM = 128
        ENC_EMB_DIM = 256
        DEC_EMB_DIM = 256
        HID_DIM = 512
        N_LAYERS = 2
        ENC_DROPOUT = 0.1
        DEC_DROPOUT = 0.1

        model = None
        #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        device = torch.device("cpu")
        print(device)

        if self.model == "Seq2SeqLSTM":
            encode = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
            decode = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
            model = Seq2SeqLSTM(encode, decode)
            print("Seq2SeqLSTM")
            print(model)

        elif self.model == "Seq2SeqBART":
            model = Seq2SeqBART(device)
            print("Seq2SeqBART")
            print(model)

        elif self.model == "Seq2SeqLSTMwizATT":
            print("Seq2SeqLSTMwizATT")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        loss_df = pd.DataFrame(columns=['Epoch', 'Iteration', 'Loss'])

        for epoch in range(self.epoch):
            model.train()
            epoch_loss = 0
            # 训练代码
            for i, batch in enumerate(self.train_iterator):
                src = batch["description"]
                trg = batch["diagnosis"]
                src_tensors = []
                trg_tensors = []

                for item in src:
                    item_no_space = item.split()
                    item_int = [int(num_str) for num_str in item_no_space]
                    item_int += [0] * (512 - len(item_int))#填充长度
                    n_src = np.asarray(item_int, dtype=int)
                    src = torch.tensor(n_src)
                    src_tensors.append(src)
                src_tensor = torch.stack(src_tensors, dim=0)#拥有src_tensor

                total_zero = 0
                for item in trg:
                    item_no_space = item.split()
                    item_int = [int(num_str) for num_str in item_no_space]

                    item_int += [0] * (128 - len(item_int))
                    total_zero = total_zero + len(item_int)
                    n_trg = np.asarray(item_int, dtype=int)
                    trg = torch.tensor(n_trg)
                    trg_tensors.append(trg)
                #print("print(item_int)", item_int)
                trg_tensor = torch.stack(trg_tensors, dim=0)
                #print("tensored src", src_tensor)
                #print("tensored src size", src_tensor.size())
                #print("tensored trg", trg_tensor)
                avg_zero = int(total_zero / 16)


                optimizer.zero_grad()



                if self.model == "Seq2SeqBART":
                    output, logits = model(src_tensor, trg_tensor)
                    summary = model.generate_summary(src_tensor, trg_tensor)
                    #print(summary)
                    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), trg_tensor.view(-1))
                    print(f'Batch {i + 1}, loss {loss.item()}')
                    loss_df = pd.concat([loss_df, pd.DataFrame({'Epoch': [epoch + 1], 'Iteration': [i + 1], 'Loss':[loss.item()]})])

                    loss.backward()
                    optimizer.step()

                else:
                    #src_tensor = torch.transpose(src_tensor, 0, 1)#gai
                    #print("trg_tensor", trg_tensor.size())
                    output = model(src_tensor, trg_tensor)
                    print("output", output)
                    print("trg_tensor", trg_tensor)
                    #output[:, -avg_zero:] = 0
                    #print("org output", output.size())
                    #output_dim = output.shape[-1]
                    #output = output.view(-1, output_dim)
                    #print("output", output.size())

                    #output = output[1:].view(-1, output_dim)#改
                    #trg_tensor = trg_tensor[1:].reshape(-1)
                    trg_tensor = trg_tensor.float()#gai
                    #print("tensored trg size", trg_tensor.size())
                    #output = output[1:].view(-1, output_dim)
                    #trg_tensor = trg_tensor[1:].view(-1)
                    #print("trg_tensor", trg_tensor)
                    loss = criterion(output, trg_tensor)
                    loss.requires_grad_(True)

                    print(f'Batch {i + 1}, loss {loss.item()}')

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()
                    epoch_loss += loss.item()
                    loss_df = pd.concat([loss_df, pd.DataFrame({'Epoch': [epoch + 1], 'Iteration': [i + 1], 'Loss':[loss.item()]})])

            if self.model == "Seq2SeqBART":
                torch.save(model.state_dict(), 'codebart.pt')#BART训练很慢，每轮训练都保存模型

            print(f'Epoch {epoch + 1}/{self.epoch}, Epoch_loss {epoch_loss / len(self.train_iterator)}')

            current_dir = os.getcwd()
            print(current_dir)
            model_name = self.model
            csv_path = os.path.join(current_dir, f'{model_name}_loss.csv')
            loss_df.to_csv(csv_path, index=False)#保存DataFrame到CSV文件

            model.eval()
            epoch_loss = 0.0

            with torch.no_grad():
                for i, batch in enumerate(self.val_iterator):
                    src = batch["description"]
                    print("origin_src", src)
                    trg = batch["diagnosis"]
                    print("origin_trg", trg)

                    src_tensors = []
                    trg_tensors = []

                    for item in src:
                        item_no_space = item.split()
                        item_int = [int(num_str) for num_str in item_no_space]
                        item_int += [0] * (512 - len(item_int))
                        n_src = np.asarray(item_int, dtype=int)
                        src = torch.tensor(n_src)
                        src_tensors.append(src)
                    src_tensor = torch.stack(src_tensors, dim=0)

                    for item in trg:
                        item_no_space = item.split()
                        item_int = [int(num_str) for num_str in item_no_space]
                        item_int += [0] * (128 - len(item_int))
                        n_trg = np.asarray(item_int, dtype=int)
                        trg = torch.tensor(n_trg)
                        trg_tensors.append(trg)
                    trg_tensor = torch.stack(trg_tensors, dim=0)

                    if self.model == "Seq2SeqBART":
                        output, logits = model(src_tensor, trg_tensor)
                        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), trg_tensor.view(-1))
                        epoch_loss += loss.item()
                        print(f'Batch {i + 1}, loss {loss.item()}')
                        #loss.backward()
                        #optimizer.step()

                    else:
                        output = model(src_tensor, trg_tensor)
                        output_dim = output.shape[-1]
                        output = output.view(-1, output_dim)
                        trg_tensor = trg_tensor.float()

                        loss = criterion(output, trg_tensor)
                        print(f'Batch {i + 1}, loss {loss.item()}')
                        epoch_loss += loss.item()
                        loss_df = pd.concat([loss_df, pd.DataFrame({'Epoch': [epoch + 1], 'Iteration': [i + 1]})])

            avg_loss = epoch_loss / len(self.val_iterator)

            print(f'Epoch {epoch + 1}/{self.epoch}, Avg LOSS: {avg_loss}')

    def test(self):
        torch.manual_seed(12)
        device = torch.device("cpu")
        model = None
        if self.model == "Seq2SeqBART":
            model = Seq2SeqBART(device)
            model.load_state_dict(torch.load('codebart.pt'))
            print(model)

            with open('BART_Metrics.csv', 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([
                    'R1_r', 'R1_p', 'R1_f',
                    'R2_r', 'R2_p', 'R2_f',
                    'Rl_r', 'Rl_p', 'Rl_f',
                    'BLEU1', 'BLEU2'
                ])

                for i, batch in enumerate(self.test_iterator):
                    model.eval()
                    src = batch["description"]
                    trg = batch["diagnosis"]

                    src_tensors = []
                    trg_tensors = []

                    for item in src:
                        item_no_space = item.split()
                        item_int = [int(num_str) for num_str in item_no_space]
                        item_int += [0] * (512 - len(item_int))
                        n_src = np.asarray(item_int, dtype=int)
                        src = torch.tensor(n_src)
                        src_tensors.append(src)
                    src_tensor = torch.stack(src_tensors, dim=0)

                    for item in trg:
                        item_no_space = item.split()
                        item_int = [int(num_str) for num_str in item_no_space]
                        item_int += [0] * (128 - len(item_int))
                        n_trg = np.asarray(item_int, dtype=int)
                        trg = torch.tensor(n_trg)
                        trg_tensors.append(trg)
                    trg_tensor = torch.stack(trg_tensors, dim=0)

                    output, logits = model(src_tensor, trg_tensor)
                    #loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), trg_tensor.view(-1))
                    #print(f'Batch {i + 1}, loss {loss.item()}')

                    summary = model.generate_summary(src_tensor, trg_tensor)
                    output_list = summary.tolist()
                    trg_list = trg_tensor[0].tolist()
                    output_list = [str(value) if value != 0 else '0' for value in output_list]
                    hypothesis = ', '.join(output_list)
                    trg_list = [str(value) if value != 0 else '0' for value in trg_list]
                    reference = ', '.join(trg_list)

                    #rouge = Rouge()
                    #scores = rouge.get_scores([hypothesis], [reference], avg=True)
                    #print(scores)
                    score_rouge = self.rouge_scores(hypothesis, reference)
                    score, score_2 = self.bleu_scores(hypothesis, reference)
                    batch_metric = {
                        'R1_r': score_rouge['rouge-1']['r'],
                        'R1_p': score_rouge['rouge-1']['p'],
                        'R1_f': score_rouge['rouge-1']['f'],
                        'R2_r': score_rouge['rouge-2']['r'],
                        'R2_p': score_rouge['rouge-2']['p'],
                        'R2_f': score_rouge['rouge-2']['f'],
                        'Rl_r': score_rouge['rouge-l']['r'],
                        'Rl_p': score_rouge['rouge-l']['p'],
                        'Rl_f': score_rouge['rouge-l']['f'],
                        'BLEU1': score,
                        'BLEU2': score_2
                    }

                    print(f'Batch {i + 1}, {score_rouge}, {score}, {score_2}')

                    csv_writer.writerow([
                        batch_metric['R1_r'], batch_metric['R1_p'], batch_metric['R1_f'],
                        batch_metric['R2_r'], batch_metric['R2_p'], batch_metric['R2_f'],
                        batch_metric['Rl_r'], batch_metric['Rl_p'], batch_metric['Rl_f'],
                        batch_metric['BLEU1'], batch_metric['BLEU2']
                    ])


        elif self.model == "Seq2SeqLSTM":
            print("model:Seq2SeqLSTM")
            print("请将main中#ProjectRun.train()的井号取消进行训练验证与测试")

    def bleu_scores(self, hypothesis, reference):
        hypothesis, reference = hypothesis.split(','), reference.split(',')
        len_hypothesis, len_reference = len(hypothesis), len(reference)
        score = math.exp(min(0, 1 - len_reference / len_hypothesis))
        for n in range(1, 1 + 1):
            num_matches, label_subs = 0, collections.defaultdict(int)
            for i in range(len_reference - n + 1):
                label_subs[','.join(reference[i: i + n])] += 1
            for i in range(len_hypothesis - n + 1):
                if label_subs[','.join(hypothesis[i: i + n])] > 0:
                    num_matches += 1
                    label_subs[','.join(hypothesis[i: i + n])] -= 1
            score *= math.pow(num_matches / (len_hypothesis - n + 1), math.pow(0.5, n))

        score_2 = math.exp(min(0, 1 - len_reference / len_hypothesis))
        for n in range(1, 1 + 2):
            num_matches, label_subs = 0, collections.defaultdict(int)
            for i in range(len_reference - n + 1):
                label_subs[','.join(reference[i: i + n])] += 1
            for i in range(len_hypothesis - n + 1):
                if label_subs[','.join(hypothesis[i: i + n])] > 0:
                    num_matches += 1
                    label_subs[','.join(hypothesis[i: i + n])] -= 1
            score_2 *= math.pow(num_matches / (len_hypothesis - n + 1), math.pow(0.5, n))
        return score, score_2

    def rouge_scores(self, hypothesis, reference):
        #print("rouge_scores")
        rouge = Rouge()
        scores = rouge.get_scores([hypothesis], [reference], avg=True)

        return scores


if __name__ == "__main__":
    ProjectRun = ProjectRun()
    ProjectRun.dataset()
    start_time = time.time()
    ProjectRun.train()
    ProjectRun.test()