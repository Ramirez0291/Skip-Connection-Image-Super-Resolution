import time

import numpy as np
import tensorflow as tf
import os
import cv2 as cv
import models
from configs import *


def main():
    assert os.path.exists(ORIGINAL_TEST_PATH)
    image_files = os.listdir(ORIGINAL_TEST_PATH)
    assert len(image_files) > 0

    run(image_files)


def run(names):

    print('load check point now~')
    ckpt_state = tf.train.get_checkpoint_state(CHECKPOINTS_PATH)
    if not ckpt_state or not ckpt_state.model_checkpoint_path:
        print('No check point files are found!')
    num_ckpt = -1
    try:
        ckpt_files = ckpt_state.all_model_checkpoint_paths
        num_ckpt = len(ckpt_files)
    except:
        pass

    if num_ckpt < 1:
        print('No check point files are found!')

    low_res_holder = tf.placeholder(
        tf.float32, shape=[1, INPUT_SIZE_RUN, INPUT_SIZE_RUN, NUM_CHENNELS])
    inferences = models.init_model(MODEL_NAME, low_res_holder)

    sess = tf.Session()
    # 全局变量的初始化
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(tf.global_variables())
    print('check point load finished')
    print('MODLE:%s' % MODEL_NAME)
    print(ckpt_files[-1])
    saver.restore(sess, ckpt_files[-1])  # 加载最后训练的模型

    for name in names:
        #try:
        print(name)
        true_res_img = cv.imread('./test/Original/' + name)
        #用作测试
        #'''
        low_res_img = cv.resize(
            true_res_img, (true_res_img.shape[1] // SCALE_FACTOR,
                           true_res_img.shape[0] // SCALE_FACTOR),
            interpolation=cv.INTER_AREA)
        #
        #实际使用
        #low_res_img = true_res_img
        filename = './test/' + MODEL_NAME + '/' + name
        print(filename)
        # low_res_img = cv.imread('C:/Users/Rami/OneDrive/画/素材/IMG_2791.JPG')
        # low_res_img = cv.resize(small_img, (small_img.shape[1]*2, small_img.shape[0]*2), interpolation=cv.INTER_CUBIC)

        # 获得图片尺寸数据与色彩通道
        img_rows = low_res_img.shape[0]
        img_cols = low_res_img.shape[1]
        img_chns = low_res_img.shape[2]
        timea = time.time()
        output_size = int(inferences.get_shape()[1])
        input_size = INPUT_SIZE_RUN
        available_size = output_size // SCALE_FACTOR
        margin = (input_size - available_size) // 2

        padded_rows = int(
            img_rows / available_size + 1) * available_size + margin * 2
        padded_cols = int(
            img_cols / available_size + 1) * available_size + margin * 2
        padded_low_res_img = np.zeros(
            (padded_rows, padded_cols, img_chns), dtype=np.uint8)
        padded_low_res_img[margin:margin + img_rows, margin:margin + img_cols,
                           ...] = low_res_img
        padded_low_res_img = padded_low_res_img.astype(np.float32)
        padded_low_res_img /= 255
        # padded_low_res_img -= 0.5

        high_res_img = np.zeros(
            (padded_rows * SCALE_FACTOR, padded_cols * SCALE_FACTOR, img_chns),
            dtype=np.float32)
        low_res_patch = np.zeros(
            (1, input_size, input_size, img_chns), dtype=np.float32)
        for i in range(margin, margin + img_rows, available_size):
            for j in range(margin, margin + img_cols, available_size):
                low_res_patch[0, ...] = padded_low_res_img[
                    i - margin:i - margin + input_size, j - margin:
                    j - margin + input_size, ...]
                high_res_patch = sess.run(
                    inferences, feed_dict={low_res_holder: low_res_patch})

                out_rows_begin = (i - margin) * SCALE_FACTOR
                out_rows_end = out_rows_begin + output_size
                out_cols_begin = (j - margin) * SCALE_FACTOR
                out_cols_end = out_cols_begin + output_size
                high_res_img[out_rows_begin:out_rows_end, out_cols_begin:
                             out_cols_end, ...] = high_res_patch[0, ...]

        # high_res_img += 0.5

        high_res_img = tf.image.convert_image_dtype(high_res_img, tf.uint8,
                                                    True)

        timename = time.time()
        print('used time:%d' % (timename - timea))

        high_res_img = high_res_img[:SCALE_FACTOR * img_rows, :
                                    SCALE_FACTOR * img_cols, ...]
        cv.imwrite(
            filename,
            high_res_img.eval(session=sess),
            [int(cv.IMWRITE_PNG_COMPRESSION), 0])

        print('Enhance Finished!')
    # except:
    #     print('ERROR!!')


if __name__ == '__main__':
    main()
