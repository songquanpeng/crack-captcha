import argparse
import io
import os
import time

import requests
from PIL import Image
from tqdm import tqdm


def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    for i in tqdm(range(args.start_num, args.end_num)):
        res = requests.get(args.target_link)
        raw = res.content
        if not raw.startswith(b'GIF89a'):
            print('Invalid response.')
            continue
        img = Image.open(io.BytesIO(raw))
        img.save(os.path.join(args.save_path, f"{i}.gif"))
        time.sleep(args.crawl_interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./archive/captcha')
    parser.add_argument('--start_num', type=int, default=0)
    parser.add_argument('--end_num', type=int, default=1000)
    parser.add_argument('--crawl_interval', type=float, default=2)
    parser.add_argument('--target_link', type=str, default='http://dyxt.jw.scut.edu.cn/CreateImg.aspx')
    main(parser.parse_args())
