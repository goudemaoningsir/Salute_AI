import os


def read_data_nmt():
    """载入“英语－法语”数据集"""
    with open(os.path.join("fra-eng", "fra.txt"), "r", encoding="utf-8") as f:
        return f.read()


raw_text = read_data_nmt()
print(raw_text[:75])


# @save
def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""

    def no_space(char, prev_char):
        return char in set(",.!?") and prev_char != " "

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace("\u202f", " ").replace("\xa0", " ").lower()
    # 在单词和标点符号之间插入空格
    out = [
        " " + char if i > 0 and no_space(char, text[i - 1]) else char
        for i, char in enumerate(text)
    ]
    return "".join(out)


text = preprocess_nmt(raw_text)
print(text[:80])
