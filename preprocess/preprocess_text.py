import argparse
import json
import logging
import os
from typing import List

import numpy as np
from sacremoses import MosesTokenizer


os.environ['CUDA_VISIBLE_DEVICES'] = "0," # 指定所用的显卡
TOKENIZER = MosesTokenizer(lang='en')  # BPE编码
origin_dir = "D:\\程序设计-python\\OpenViDial\\MyOpenViDial\\origin_data"
output_dir = "D:\\程序设计-python\\OpenViDial\\MyOpenViDial\\data"
sent_num_file = "sent_num.npy"
offset_file = "offset.npy"
src_file = "src.txt"

# 加载text，得到sentence-sample-samples output
def load_origin_texts(data_dir, split="train") -> List[List[str]]:
    """load origin text data"""
    output = []
    ori_sen = []
    input_path = os.path.join(data_dir, f'{split}.origin.txt')
    logging.info(f"Loading origin data from {input_path}")
    with open(input_path, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            line = line.replace("\u2013", "-").replace("&apos;", "'")  
            # \u2013是一个Unicode转义序列，代表了一个特殊的长横线字符
            ori_sen.append(line)
        f.close()
    
    input_path = os.path.join(data_dir, f'{split}.dialogue.jsonl')
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()  # 去除行首尾的空白字符
            if not line:
                continue
            ids = json.loads(line)  # 将当前行解析为一个JSON对象

            t_list = []
            for id_ in ids:
                t_list.append(ori_sen[id_])
            output.append(t_list)
        f.close()
    logging.info(f"Loaded {sum(len(x) for x in output)} sentences from {os.path.join(data_dir, f'{split}.origin.txt')}")
    return output
    

# 根据给定的图像目录、数据集划分和对话句子数量，获取每个句子对应的图像路径，并将这些路径存储在一个列表中并返回
def iterate_imgs(img_dir, split, sent_num: np.array) -> List[str]:
    """get image-paths according to sent-num array"""
    ids = []
    input_path = os.path.join(img_dir, f'{split}.dialogue.jsonl')
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(json.loads(line))
        f.close()
    
    output = []
    for group_idx in range(sent_num.shape[0]):
        for sent_idx in range(sent_num[group_idx]):
            output.append(os.path.join(img_dir, f"{split}_images", f"{ids[group_idx][sent_idx]}.jpg"))
    return output


def tokenize_text(texts: List[str]):
    return [TOKENIZER.tokenize(t, return_str=True) for t in texts]


def main():
    parser = argparse.ArgumentParser(description='video-data pre-processing.')
    # 命令行解析器，用于处理命令行参数，可以方便地在命令行界面中接收和解析用户输入的参数

    parser.add_argument('--origin-dir', default=origin_dir,
                        help='origin data directory.')
    parser.add_argument('--output-dir', default=output_dir,
                        help='output directory.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size when processing image')
    parser.add_argument('--workers', type=int, default=32,
                        help='cpu workers')
    parser.add_argument('--max_sent', type=int, default=5,
                        help='max history sentence number in src')
    parser.add_argument('--split', type=str, default="train",
                        help='split of dataset, train/valid/test')
    args = parser.parse_args()  # 解析用户输入的参数

    os.makedirs(args.output_dir, exist_ok=True)

    # Load text
    group_texts = load_origin_texts(args.origin_dir, args.split)

    # tokenize text
    with open(os.path.join(args.output_dir, src_file), "w", encoding='utf-8') as fsrc:
        for group_idx, group_text in enumerate(group_texts):
            tokenized_group = tokenize_text(group_text)
            for src in tokenized_group:
                fsrc.write(src + "\n")

    # compute group offsets/nums
    sent_num = np.array([len(g) for g in group_texts])
    sent_cumsum = np.cumsum(sent_num)
    offsets = np.insert(sent_cumsum[: -1], obj=0, values=0)
    np.save(os.path.join(args.output_dir, sent_num_file), sent_num)
    np.save(os.path.join(args.output_dir, offset_file), offsets)
    print(f"Moses tokenization and offsets computing Finished.")


if __name__ == '__main__':
    main()
