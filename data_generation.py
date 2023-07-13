import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, IterableDataset
from typing import Iterator, List
import sys 
import os
from preprocess.generate_dict import Vocab
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
data_dir = "D:\\程序设计-python\\OpenViDial\\MyOpenViDial\\data"

dic_file = "dic.txt"
sent_num_file = "sent_num.npy"
offset_file = "offset.npy"
feature_file = "resnet50_feature.mmap"
txt_file = "src.txt"
img_idx_file = "img_idx.npy"

NUM = 100


class FeatureDataset(IterableDataset):
    """Load Feature dataset"""
    def __init__(self, data_dir, dtype):
        super().__init__()
        self.data_dir = data_dir
        self.sent_num = np.load(os.path.join(data_dir, sent_num_file))
        self.offsets = np.load(os.path.join(data_dir, offset_file))

        self.dim = 1000
        self.total_num = self.offsets[-1] + self.sent_num[-1]
        
        self.dtype = dtype
        self.filesize = os.path.getsize(os.path.join(data_dir, feature_file))

    def __getitem__(self, index):
        offset = index * np.dtype(self.dtype).itemsize * self.dim
        assert offset < self.filesize
        arr = np.memmap(os.path.join(self.data_dir, feature_file), 
                        dtype=self.dtype, mode='r', shape=self.dim, offset=offset, order='C')
        return np.array(arr)


class TextDataset(Dataset):
    def __init__(self, datas, vocab) -> None:
        super().__init__()
        self.data = [vocab[sentence] for sentence in datas]

    def __getitem__(self, index):
        return self.data[index]


class TextImageDataset(Dataset):
    def __init__(self, image_dataset, text_dataset, vocab, mapping, span_idxs, shuffle=False):
        super().__init__()
        self.img_dataset = image_dataset
        self.text_dataset = text_dataset
        self.vocab = vocab
        self.mapping = mapping
        self.span_idxs = span_idxs
        self.shuffle = shuffle

    def __getitem__(self, index):
        group_idx, start_idx, end_idx = self.span_idxs[index].tolist()
        # 得到每个句子在txt中的位置
        offsets = [self.get_1doffsets(group_idx, sent_idx) for sent_idx in range(start_idx, end_idx+1)]
        source_imgs = np.stack([self.img_dataset[self.mapping[idx]] for idx in offsets])  # n * dim
        source_texts = np.concatenate([[self.vocab['<sep>']] + self.text_dataset[idx] for idx in offsets[:-1]])  # L
        source_texts[0] = self.vocab['<cls>']
        target = [self.vocab['<cls>']] + self.text_dataset[offsets[-1]] + [self.vocab['<sep>']]

        return (
            torch.FloatTensor(source_imgs),
            torch.LongTensor(source_texts),
            torch.LongTensor(target))

    def __len__(self):
        return len(self.span_idxs)

    def get_1doffsets(self, group_idx, sent_idx):
        group_offset = int(self.img_dataset.offsets[group_idx - 1])
        sent_num = int(self.img_dataset.sent_num[group_idx - 1])
        assert sent_idx < sent_num, f"origin text group {group_idx} has {sent_num} sents, " \
                                    f" sent_idx {sent_idx} should be less than {sent_num}"
        return group_offset + sent_idx



def item2span_idxs(sent_num: np.array, max_src_sent: int) -> np.array:
    """
    compute each src/tgt span of dataset.
    For example, if we got [[0,1,2], [3,4]] as source texts,
    sent_num should be [3, 2], and we want to use only one sentence as src.
    the output should be [[0, 0, 1], [0, 1, 2], [1, 0, 1]]
    """
    span_idxs = []
    for group_idx in range(sent_num.shape[0]):
        num = int(sent_num[group_idx])
        for sent_idx in range(1, num):  # predict texts[i] given texts[:i]
            start_idx = max(0, sent_idx - max_src_sent)
            span_idxs.append((group_idx + 1, start_idx, sent_idx))
    return np.array(span_idxs)


def get_data(max_src_sent=2, batch_size = 32):
    with open(os.path.join(data_dir, dic_file), mode='r', encoding='utf-8') as f:
        tokens = f.readlines()
        tokens = [token.strip() for token in tokens]
    vocab_dict = Vocab(tokens, load=True)

    with open(os.path.join(data_dir, txt_file), mode='r', encoding='utf-8') as f:
        sentences = f.readlines()
        sentences = [sentence.strip().split() for sentence in sentences]
    text_dataset = TextDataset(sentences, vocab_dict)
    features_dataset = FeatureDataset(data_dir=data_dir, dtype=np.float32)

    dialogue_idx = set(np.load(os.path.join(data_dir, "dialogue_idx.npy")))
    dialogue = np.load(os.path.join(data_dir, "dialogue.npy"), allow_pickle=True)
    dialogue = [id for tem in dialogue for id in tem]
    mapping = {dialogue_idx: img_idx for img_idx, dialogue_idx in enumerate(dialogue)}
    # print(dialogue_idx)
    # print(dialogue[:10])

    span_idxs = item2span_idxs(sent_num=features_dataset.sent_num,
                                    max_src_sent=max_src_sent)

    span_idxs = [span_idx for span_idx in span_idxs if span_idx[0] in dialogue_idx]
    # print(span_idxs)

    data = TextImageDataset(text_dataset=text_dataset,
                            image_dataset=features_dataset,
                            vocab=vocab_dict,
                            mapping=mapping,
                            span_idxs=span_idxs)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4)

    """for src_img, src_tokens, tgt_tokens in dataloader:
        for src_token, tgt_token in zip(src_tokens, tgt_tokens):
            print(vocab_dict.idx2token(src_token))
            print(vocab_dict.idx2token(tgt_token))
        break"""

    return dataloader, vocab_dict


def collate_fn(batch):
    # 对输入进行填充，使其具有相同的长度
    src_img = [item[0] for item in batch]
    src_tokens = [item[1] for item in batch]
    
    padded_src_tokens = my_pad_sequence(src_tokens)

    tgt_tokens = [item[2] for item in batch]
    padded_tgt_tokens = my_pad_sequence(tgt_tokens)

    return (src_img, padded_src_tokens, padded_tgt_tokens)

def my_pad_sequence(sequences):
        max_len = max([seq.size(0) for seq in sequences])
        padded_sequences = [F.pad(seq, (0, max_len - seq.size(0)), value=3) for seq in sequences]
        return torch.stack(padded_sequences)

if __name__ == "__main__":
    dataloader = get_data(max_src_sent=5, batch_size=2)
    

    