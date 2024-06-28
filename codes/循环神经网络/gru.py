import collections
import math
import requests
import torch
from torch import nn
import torch.nn.functional as F
import random


def download(url, filepath):
    response = requests.get(url)
    with open(filepath, "w") as f:
        f.write(response.text)


# 下载数据集
url = "https://www.gutenberg.org/files/35/35-0.txt"
filepath = "timemachine.txt"
download(url, filepath)


# 读取《时间机器》文本数据
def read_time_machine():
    with open("timemachine.txt", "r") as f:
        lines = f.readlines()
    return [line.strip().lower() for line in lines]


# 构建字符索引字典
class Vocab:
    def __init__(self, tokens, min_freq=0, reserved_tokens=None):
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元'<unk>'的索引是0
        self.idx_to_token = ["<unk>"] + reserved_tokens
        self.idx_to_token += [
            token
            for token, freq in self.token_freqs
            if freq >= min_freq and token not in self.idx_to_token
        ]
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.token_to_idx["<unk>"])
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


# 生成数据迭代器
class SeqDataLoader:
    def __init__(self, seqs, vocab, batch_size, num_steps):
        self.vocab = vocab
        self.data = self.preprocess(seqs, vocab)
        self.batch_size, self.num_steps = batch_size, num_steps
        self.num_batches = (len(self.data) - 1) // (batch_size * num_steps)

    def preprocess(self, seqs, vocab):
        return [vocab[char] for char in seqs]

    def __iter__(self):
        Xs, Ys = [], []
        for i in range(0, len(self.data) - 1, self.num_steps):
            Xs.append(self.data[i : i + self.num_steps])
            Ys.append(self.data[i + 1 : i + 1 + self.num_steps])
        for i in range(0, len(Xs) - self.batch_size + 1, self.batch_size):
            X = torch.tensor(Xs[i : i + self.batch_size])
            Y = torch.tensor(Ys[i : i + self.batch_size])
            yield X, Y


def load_data_time_machine(batch_size, num_steps):
    seqs = read_time_machine()
    vocab = Vocab(seqs)
    return SeqDataLoader(seqs, vocab, batch_size, num_steps), vocab


# 加载数据
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)


class GRUModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, device):
        super(GRUModel, self).__init__()
        self.num_hiddens = num_hiddens
        self.gru = nn.GRU(vocab_size, num_hiddens)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        self.device = device

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), len(vocab)).float().to(self.device)
        Y, state = self.gru(X, state)
        output = self.dense(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, batch_size):
        return torch.zeros((1, batch_size, self.num_hiddens), device=self.device)


def grad_clipping(net, theta):
    params = [p for p in net.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum(p.grad**2) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_ch8(net, train_iter, vocab, lr, num_epochs, device):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.to(device)
    for epoch in range(num_epochs):
        state = None
        for X, Y in train_iter:
            if state is None:
                state = net.begin_state(batch_size=X.shape[0])
            else:
                if isinstance(state, (tuple, list)):
                    for s in state:
                        s.detach_()
                else:
                    state.detach_()
            X, Y = X.to(device), Y.to(device)
            y_hat, state = net(X, state)
            y = Y.T.reshape(-1)
            l = loss(y_hat, y.long()).mean()
            optimizer.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            optimizer.step()
        print(f"epoch {epoch + 1}, loss {l.item():.3f}")


# 训练模型
num_epochs, lr = 500, 1e-3
num_hiddens = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = GRUModel(len(vocab), num_hiddens, device)
train_ch8(net, train_iter, vocab, lr, num_epochs, device)


def predict_ch8(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return "".join([vocab.idx_to_token[i] for i in outputs])


# 预测结果
print(predict_ch8("time traveller", 50, net, vocab, device))
