import os
import numpy as np

SPLIT_RATIO = [0.4, 0.4, 0.1, 0.1]
TRAIN_DIR = ["t2", "t1ce"]
LABEL_DIR = ["segs"]
ROOT = "/home/featurize/data/BraTS2D"
LABLE_DIR = os.path.join(ROOT, LABEL_DIR[0])
LABEL_PATH = os.listdir(LABLE_DIR)
MAX_len = len(os.listdir(os.path.join(ROOT, TRAIN_DIR[0])))
np.random.seed(666)
ALL_labels = np.random.permutation(MAX_len)
TRAIN_PART1_NUM, TRAIN_PART2_NUM, VAL_NUM, TEST_NUM = int(SPLIT_RATIO[0]*MAX_len), int(sum(SPLIT_RATIO[:2])*MAX_len),\
                                                      int(sum(SPLIT_RATIO[:3])*MAX_len), int(sum(SPLIT_RATIO)*MAX_len)
print(f"MAX_category: {MAX_len}, train_part1: {TRAIN_PART1_NUM}, train_part2: {TRAIN_PART2_NUM-TRAIN_PART1_NUM}, "
      f"val: {VAL_NUM-TRAIN_PART2_NUM}, test: {TEST_NUM-VAL_NUM}")
for train_dir in TRAIN_DIR: 
    cur_dir = os.path.join(ROOT, train_dir)
    cur_imgs = os.listdir(cur_dir)
    f_train_part1 = open(os.path.join(ROOT, train_dir + "_part1_train.txt"), "w")
    f_train_part2 = open(os.path.join(ROOT, train_dir + "_part2_train.txt"), "w")
    f_val = open(os.path.join(ROOT, train_dir + "_val.txt"), "w")
    f_test = open(os.path.join(ROOT, train_dir + "_test.txt"), "w")
    for i in range(MAX_len):
        cur_index = ALL_labels[i]
        img_path = os.path.join(cur_dir, cur_imgs[cur_index])
        label_path = os.path.join(LABLE_DIR, LABEL_PATH[cur_index])
        imgs = sorted(os.listdir(img_path))
        labels = sorted(os.listdir(label_path))
        for j in range(len(imgs)):
            img = os.path.join(img_path, imgs[j])
            label = os.path.join(label_path, labels[j])
            write_label = img + " " + label + "\n"
            if i < TRAIN_PART1_NUM:
                f_train_part1.write(write_label)
            elif i < TRAIN_PART2_NUM:
                f_train_part2.write(write_label)
            elif i < VAL_NUM:
                f_val.write(write_label)
            else:
                f_test.write(write_label)
    f_train_part1.close()
    f_train_part2.close()
    f_val.close()
    f_test.close()
