import torch
import torch.nn as nn
import random
import torch.nn.functional as F

class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=1):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        #outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)
        outputs = torch.zeros(batch_size, trg_vocab_size)
        hidden, cell = self.encoder(src) #encode的last hidden状态->decode的initial

        input = trg[:, 0]
        #print("input", input)

        for t in range(1, trg_len):#迭代生成词
            output, hidden, cell = self.decoder(input, hidden, cell)
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) #要概率最大的那个
            outputs[:, t - 1] = top1
            input = trg[:, t - 1] if teacher_force else top1 #要teaching的话去grand Truth的第t个
        return outputs

# 定义Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hidden_dim
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.emb_dim = emb_dim

        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)#拿到LSTM的最后一层输出,最后一步的隐藏状态
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        self.embedding = nn.Embedding(max(self.input_dim, torch.max(src).item() + 1), self.emb_dim)
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        outputs, (hidden, cell) = self.rnn(embedded)
        # hidden = [num_layers, batch_size, hidden]
        #print("outputs", outputs.size())
        return hidden, cell

# 定义Decoder
class Decoder(nn.Module):
    # def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout, attention):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.emb_dim = emb_dim

        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.log_softmax = nn.LogSoftmax()

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        # self.attention = attention

    def forward(self, input, hidden, cell):
        self.embedding = nn.Embedding(max(self.output_dim, torch.max(input).item() + 1), self.emb_dim)
        input = input.unsqueeze(1)
        #print(input.size())
        #input [batch_size, 1] ,匹配维度
        embedded = self.dropout(self.embedding(input))
        # a = self.attention(hidden, encoder_outputs)
        # a = a.unsqueeze(1)
        # encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # weighted = torch.bmm(a, encoder_outputs)
        # weighted = weighted.permute(1, 0, 2)
        # rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = output.squeeze(0)
        # weighted = weighted.squeeze(0)
        # embedded = embedded.squeeze(0)
        # prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell