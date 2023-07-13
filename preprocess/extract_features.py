import argparse
import os
import warnings
from typing import List
import json
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

CNN = torchvision.models.resnet50(pretrained=True)
CNN.eval()
CNN_FEATURE_DIM = 1000
if torch.cuda.is_available():
    CNN = CNN.cuda()
else:
    pass
    # warnings.warn("cuda not available")

origin_dir = "D:\程序设计-python\OpenViDial\MyOpenViDial\origin_data"
output_dir = "D:\程序设计-python\OpenViDial\MyOpenViDial\data"
feature_file = "resnet50_feature.mmap"

class ImageDataset(Dataset):
    def __init__(self, file_names: List[str]):
        self.file_names = file_names
        self.preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=Image.LINEAR),
            transforms.CenterCrop(224),  # todo CenterCrop可能会漏掉长屏幕的信息，待优化
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, item):
        return self.preprocess(Image.open(self.file_names[item]))


def get_dataloader(file_names, batch_size, workers):
    dataset = ImageDataset(file_names)
    return DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )


def main():
    parser = argparse.ArgumentParser(description='extract CNN pooling features from images')

    parser.add_argument('--origin-dir', default=origin_dir,
                        help='origin data directory.')
    parser.add_argument('--output-dir', default=output_dir,
                        help='output directory.')
    parser.add_argument('--split', type=str, default="train",
                        help='split of dataset, train/valid/test')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size when processing image')
    parser.add_argument('--workers', type=int, default=8,
                        help='cpu workers')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ids = np.load(os.path.join(output_dir, "dialogue.npy"), allow_pickle=True)
    ids = [img for idx in ids for img in idx]
    template = "D:\\程序设计-python\\OpenViDial\\MyOpenViDial\\origin_data\\train_images\\{}.jpg"
    image_files = [template.format(str(idx)) for idx in ids]

    total_num = len(image_files)
    print(f"Computing {CNN_FEATURE_DIM}-dim features for {total_num} pictures")

    feature_map = np.memmap(os.path.join(args.output_dir, feature_file), dtype='float32', mode='w+',
                            shape=(total_num, CNN_FEATURE_DIM))
    idx = 0
    img_dataloder = get_dataloader(file_names=image_files,
                                   batch_size=args.batch_size,
                                   workers=args.workers)

    for input_batch in tqdm(img_dataloder):  # 显示进度条
        with torch.no_grad():
            features = CNN(input_batch).numpy()
        img_num = features.shape[0]
        for img_idx in range(img_num):
            feature_map[idx + img_idx] = features[img_idx]
        idx += img_num


if __name__ == '__main__':
    main()
   


