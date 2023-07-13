import os
import json
import numpy as np

origin_dir = "D:\\程序设计-python\\OpenViDial\\MyOpenViDial\\origin_data"
output_dir = "D:\程序设计-python\OpenViDial\MyOpenViDial\data"

output = []
for idx in os.listdir(os.path.join(origin_dir, "train_images")):
    output.append(int(idx.split('.')[0]))
output = sorted(output)
img_set = set(output)

ids = []
with open(os.path.join(origin_dir, "train.dialogue.jsonl")) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        idx = json.loads(line)
        ids.append(idx)


output_ids = []
nums = []
num = 1
for line in ids:
    if set(line) < img_set:
        output_ids.append(line)
        nums.append(num)
    num += 1


np.save(os.path.join(output_dir, "dialogue.npy"), output_ids)
np.save(os.path.join(output_dir, "dialogue_idx.npy"), nums)



