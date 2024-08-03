import os
import shutil

import numpy as np

import cv2 as cv
from configs import *

BIG_DATA_PATH = './Image2X/data/big'


def clear_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        old_pairs = os.listdir(path)
        if len(old_pairs) > 0:
            shutil.rmtree(path)
            os.mkdir(path)


def main():
    #clear_dir(BIG_DATA_PATH)
    #clear_dir(ORIGINAL_IMAGES_PATH)

    assert os.path.exists(BIG_DATA_PATH)
    image_files = os.listdir(BIG_DATA_PATH)
    assert len(image_files) > 0

    image_size_change(image_files)

    print('Image changing finished!')


def image_size_change(image_files):
    file_id = 0
    for image_file in image_files:
        file_id += 1
        img = cv.imread(os.path.join(BIG_DATA_PATH, image_file))
        if img is None:
            continue
        print('oringinal image: ', image_file)

        rows = img.shape[0]
        cols = img.shape[1]

        scale = rows / 400
        new_cols = int(cols // scale)
        save_name = '%d.png' % file_id
        # 使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现。当图像放大时，类似于 CV_INTER_NN 方法
        resi = cv.resize(img, (new_cols,400), interpolation=cv.INTER_AREA)
        cv.imwrite(
            os.path.join(ORIGINAL_IMAGES_PATH, save_name), resi,
            [int(cv.IMWRITE_PNG_COMPRESSION), 0])


if __name__ == '__main__':
    main()
