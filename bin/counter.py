import argparse
import os
from collections import defaultdict

from utils.file import list_all_images


def main(args):
    image_paths = list_all_images(args.dataset_path)
    labels = [os.path.splitext(os.path.basename(path))[0] for path in image_paths]
    count = defaultdict(lambda: 0)
    for label in labels:
        assert len(label) == 4
        for c in label:
            assert c in args.alphabet
            count[c] += 1
    keys = sorted(count, key=count.get, reverse=True)
    for key in keys:
        print(f"{key}: {count[key]}")
    print("Alphabet: ", ''.join(sorted(keys)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./archive/captcha')
    parser.add_argument('--alphabet', type=str, default='02468BDFHJLNPRTVXZ')
    main(parser.parse_args())
