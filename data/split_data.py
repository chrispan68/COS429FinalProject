import os
import sys
import argparse
import random
from shutil import copyfile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Data folder")
    args = parser.parse_args()

    random.seed(100)

    for label in ['glaucoma', 'healthy']:
        data_dir = os.path.join(args.data_dir, "data/" + label)
        train_dir = os.path.join(args.data_dir, "train/" + label)
        valid_dir = os.path.join(args.data_dir, "val/" + label)
        test_dir = os.path.join(args.data_dir, "test/" + label)
        
        split = [0.6, 0.8, 1.0]

        imgs = []
        for file in os.listdir(data_dir):
            if file.endswith('png') or file.endswith('jpg'):
                imgs.append(file)
        
        random.shuffle(imgs)

        split = [int(a * len(imgs)) for a in split]

        train = imgs[:split[0]]
        valid = imgs[split[0]:split[1]]
        test = imgs[split[1]:]
        for file in train:
            copyfile(os.path.join(data_dir, file), os.path.join(train_dir, file))
        for file in valid:
            copyfile(os.path.join(data_dir, file), os.path.join(valid_dir, file))
        for file in test:
            copyfile(os.path.join(data_dir, file), os.path.join(test_dir, file))
        
if __name__ == "__main__":
    main()