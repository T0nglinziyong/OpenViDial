import collections
import os

dir = "D:\程序设计-python\OpenViDial\MyOpenViDial\data"
class Vocab:
    def __init__(self, datas=None, min_freq=0, reserved_tokens=None, load=False):
        if load is True:
            self.idx_to_token = datas
        else:
            if reserved_tokens is None:
                reserved_tokens = []
            counter_dic = count(datas)
            token_freq = sorted(counter_dic.items(), key=lambda x: x[1], reverse=True)
            token_lst = [token for token, freq in token_freq if freq > min_freq]
            self.idx_to_token = ['<unk>'] + reserved_tokens + token_lst
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __getitem__(self, tokens):
        if isinstance(tokens, (list, tuple)):
            return [self.token_to_idx.get(token, self.unk) for token in tokens]
        else:
             return self.token_to_idx.get(tokens, self.unk)

    def __len__(self):
        return len(self.idx_to_token)

    def idx2token(self, ids):
        try:
            return [self.idx_to_token[idx] for idx in ids]
        except:
             return self.idx_to_token[ids]

    @property
    def unk(self):
        return 0


def count(datas):
    if isinstance(datas[0], (list, tuple)):
        datas = [token for data in datas for token in data]
    return collections.Counter(datas)

if __name__ == "__main__":
    sentences = []
    with open(os.path.join(dir, "src.txt") , mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        sentences.append(line)
    vocab = Vocab(sentences, min_freq=5, reserved_tokens=['<cls>', '<sep>', '<pad>'])
    with open(os.path.join(dir, "dic.txt"), mode='w', encoding='utf-8') as f:
        for i in range(len(vocab)):
            f.write(vocab.idx2token(i) + '\n')
        