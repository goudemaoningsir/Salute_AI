import torch
import random
import re
from collections import Counter
from torch.utils.data import DataLoader, Dataset


# 读取文本文件并进行简单预处理
def read_text_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    return [re.sub("[^A-Za-z]+", " ", line).strip().lower() for line in lines]


# 词元化文本
def tokenize(lines):
    return [line.split() for line in lines]


# 构建词表
class Vocab:
    def __init__(self, tokens, min_freq=0, reserved_tokens=None):
        if reserved_tokens is None:
            reserved_tokens = []
        counter = Counter([token for line in tokens for token in line])
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ["<unk>"] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self.token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

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


# 序列数据集
class SeqDataset(Dataset):
    def __init__(self, corpus, vocab, num_steps):
        self.corpus = [vocab[token] for line in corpus for token in line]
        self.num_steps = num_steps

    def __len__(self):
        return (len(self.corpus) - 1) // self.num_steps

    def __getitem__(self, idx):
        start = idx * self.num_steps
        end = start + self.num_steps
        X = self.corpus[start:end]
        Y = self.corpus[start + 1 : end + 1]
        return torch.tensor(X), torch.tensor(Y)


# 生成批量数据
def load_data(corpus, vocab, batch_size, num_steps):
    dataset = SeqDataset(corpus, vocab, num_steps)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 主程序
if __name__ == "__main__":
    filepath = "timemachine.txt"
    lines = read_text_file(filepath)
    tokens = tokenize(lines)
    vocab = Vocab(tokens)

    print(tokens)
    # 打印词频前10的单词
    print("打印词频前10的单词:", vocab.token_freqs[:10])

    # 生成数据批量
    batch_size, num_steps = 2, 5
    data_iter = load_data(tokens, vocab, batch_size, num_steps)

    # 测试生成的批量数据
    for X, Y in data_iter:
        print("X:", X)
        print("Y:", Y)
        break
