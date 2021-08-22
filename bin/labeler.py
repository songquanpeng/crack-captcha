import argparse
import os

import easyocr
import numpy as np
from PIL import Image
from tqdm import tqdm

"""

easyocr is unable to handle this.

The code is kept only for reference.

"""


def main(args):
    ocr = easyocr.Reader(['en'])
    for i in tqdm(range(args.start_num, args.end_num)):
        old_path = os.path.join(args.dataset_path, f"{i}.gif")
        assert os.path.exists(old_path)
        img = Image.open(old_path)
        img = np.array(img)
        captcha = ocr.readtext(img, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', detail=0)
        if len(captcha) != 1 or len(captcha[0]) != 4:
            print(f"Incorrect ocr for image: {old_path}")
            continue
        label = captcha[0].upper()
        new_path = os.path.join(args.dataset_path, f"{label}.gif")
        os.rename(old_path, new_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./archive/captcha/train')
    parser.add_argument('--start_num', type=int, default=302)
    parser.add_argument('--end_num', type=int, default=1000)
    main(parser.parse_args())
