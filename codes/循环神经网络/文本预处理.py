import collections
import re
import requests

print(
    "================================= 1. 读取数据集 ================================="
)


# 下载并读取数据
def download(url, filepath):
    response = requests.get(url)
    with open(filepath, "w") as f:
        f.write(response.text)


# 下载数据集
url = "https://www.gutenberg.org/files/35/35-0.txt"
filepath = "timemachine.txt"
download(url, filepath)


def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    with open(filepath, "r") as f:
        lines = f.readlines()
    # 只保留字母字符，并转成小写
    return [re.sub("[^A-Za-z]+", " ", line).strip().lower() for line in lines]


# 读取数据集并打印示例行
lines = read_time_machine()
print("Lines:", lines)
print(f"# 文本总行数: {len(lines)}")
for i in range(5):
    print(f"lines[{i}]：", lines[i])

print("================================= 2. 词元化 =================================")


def tokenize(lines, token="word"):
    """将文本行拆分为单词或字符词元"""
    if token == "word":
        return [line.split() for line in lines]
    elif token == "char":
        return [list(line) for line in lines]
    else:
        print("错误：未知词元类型：" + token)


# 词元化示例
tokens = tokenize(lines, token="word")
for i in range(5):
    print(f"{i}：", tokens[i])

print("================================= 3. 词表 =================================")


class Vocab:
    """文本词表"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ["<unk>"] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):
    """统计词元的频率"""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


# 构建词表并打印前几个高频词元及其索引
vocab = Vocab(tokens)
print("前几个高频词元及其索引:", list(vocab.token_to_idx.items())[:10])

# 将文本转换为数字索引序列的示例
for i in [0, 10]:
    print("文本:", tokens[i])
    print("索引:", vocab[tokens[i]])


def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, "char")
    vocab = Vocab(tokens)
    # 将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


# 加载语料库和词表，并打印它们的长度
corpus, vocab = load_corpus_time_machine()
print(len(corpus), len(vocab))
