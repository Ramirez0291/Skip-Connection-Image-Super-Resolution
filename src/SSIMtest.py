import math
import os

import numpy as np
import tensorflow as tf
from scipy import ndimage, signal
from scipy.ndimage.filters import convolve

import cv2 as cv
import models
from configs import *
from input_data import *


def main():
    assert os.path.exists(KODAK_TEST_SET)
    image_files = os.listdir(KODAK_TEST_SET)
    assert len(image_files) > 0
    
    test(image_files)


def test(names):
    ckpt_state = tf.train.get_checkpoint_state(CHECKPOINTS_PATH)
    if not ckpt_state or not ckpt_state.model_checkpoint_path:
        print('No check point files are found!')
        return

    ckpt_files = ckpt_state.all_model_checkpoint_paths
    num_ckpt = len(ckpt_files)
    if num_ckpt < 1:
        print('No check point files are found!')
        return


    low_res_holder = tf.placeholder(
        tf.float32,
        shape=[1, INPUT_SIZE_RUN, INPUT_SIZE_RUN, NUM_CHENNELS])
    inferences = models.init_model(MODEL_NAME, low_res_holder)

    sess = tf.Session()
    # 全局变量的初始化
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver = tf.train.Saver(tf.global_variables())
    tf.train.start_queue_runners(sess=sess)
    for i in range(30,1000):
        ckpt_file = ckpt_files[i]
        print(ckpt_file)
        saver.restore(sess, ckpt_file)

        avg_ssim = 0.0
        avg_psnr = 0.0
        for name in names:
            #try:
            print(name)
            true_res_img = cv.imread('./test/KODAK/' + name)
            #用作测试
            low_res_img = cv.resize(
                true_res_img,
                (true_res_img.shape[1] // 2, true_res_img.shape[0] // 2),
                interpolation=cv.INTER_CUBIC)

            output_size = int(inferences.get_shape()[1])
            input_size = INPUT_SIZE_RUN
            available_size = output_size // SCALE_FACTOR
            margin = (input_size - available_size) // 2

            # 获得图片尺寸数据与色彩通道0
            img_rows = low_res_img.shape[0]
            img_cols = low_res_img.shape[1]
            img_chns = low_res_img.shape[2]

            padded_rows = int(
                img_rows / available_size + 1) * available_size + margin * 2
            padded_cols = int(
                img_cols / available_size + 1) * available_size + margin * 2
            padded_low_res_img = np.zeros(
                (padded_rows, padded_cols, img_chns), dtype=np.uint8)
            padded_low_res_img[margin:margin + img_rows, margin:
                                margin + img_cols, ...] = low_res_img
            padded_low_res_img = padded_low_res_img.astype(np.float32)
            padded_low_res_img /= 255
            # padded_low_res_img -= 0.5

            high_res_img = np.zeros(
                (padded_rows * SCALE_FACTOR, padded_cols * SCALE_FACTOR,
                    img_chns),
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
            high_res_img = tf.image.convert_image_dtype(
                high_res_img, tf.uint8, True)

            #timename = time.time()
            #print('used time:%d' % (timename - timea))

            high_res_img = high_res_img[:SCALE_FACTOR * img_rows, :
                                        SCALE_FACTOR * img_cols, ...]
            #cv.imwrite(filename, high_res_img.eval(session=sess))
            avg_ssim += getMSSIM(
                np.asarray(high_res_img.eval(session=sess)),
                np.asarray(true_res_img))
            '''
            avg_ssim = MultiScaleSSIM(
                np.asarray(high_res_img.eval(session=sess)),
                np.asarray(true_res_img),
                max_val=255)
                '''
            avg_psnr += getPSNR(
                np.asarray(high_res_img.eval(session=sess)),
                np.asarray(true_res_img))
            #print('Enhance Finished!')

        print('Model: %s. MSSIM: %.3f' % (ckpt_file, avg_ssim / len(names)))
        print('Model: %s. PSNR: %.3f' % (ckpt_file, avg_psnr / len(names)))
        


def ssim(imgA,
         imgB,
         max_val=255,
         filter_size=11,
         filter_sigma=1.5,
         k1=0.01,
         k2=0.03):
    """Return the Structural Similarity Map corresponding to input images imgA 
    and imgB (images are assumed to be uint8)
    
    This function attempts to mimic precisely the functionality of ssim.m a 
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    imgA = imgA.astype(np.float32)
    imgB = imgB.astype(np.float32)
    _,height, width, _ = imgA.shape

    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(_FSpecialGauss(size, sigma), (1,size, size, 1))
        mu1 = signal.fftconvolve(imgA, window, mode='valid')
        mu2 = signal.fftconvolve(imgB, window, mode='valid')
        sigma11 = signal.fftconvolve(imgA * imgA, window, mode='valid')
        sigma22 = signal.fftconvolve(imgB * imgB, window, mode='valid')
        sigma12 = signal.fftconvolve(imgA * imgB, window, mode='valid')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = imgA, imgB
        sigma11 = imgA * imgA
        sigma22 = imgB * imgB
        sigma12 = imgA * imgB

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
    cs = np.mean(v1 / v2)
    return ssim, cs


def getMSSIM(imgA,
             imgB,
             max_val=255,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             weights=None):
    """
    This function implements Multi-Scale Structural Similarity (MSSSIM) Image 
    Quality Assessment according to Z. Wang's "Multi-scale structural similarity 
    for image quality assessment" Invited Paper, IEEE Asilomar Conference on 
    Signals, Systems and Computers, Nov. 2003 

    Author's MATLAB implementation:-
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    """
    imgA = imgA[np.newaxis,:]
    imgB = imgB[np.newaxis,:]
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((1,2, 2, 1)) / 4.0
    imgA = imgA.astype(np.float32)
    imgB = imgB.astype(np.float32)
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map = ssim(
            imgA,
            imgB,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2)
        mssim = np.append(mssim, ssim_map)
        mcs = np.append(mcs, cs_map)
        filtered = [ndimage.filters.convolve(im, downsample_filter, mode='reflect')
                for im in [imgA, imgB]]
        imgA, imgB = [x[:, ::2, ::2, :] for x in filtered]
        
    return (np.prod(mcs[0:level - 1]**weight[0:level - 1]) *
            (mssim[level - 1]**weight[level - 1]))


def getPSNR(imgA, imgB):
    mse = np.mean((imgA - imgB)**2)
    if mse == 0:
        return 0
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


#!/usr/bin/python
#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Python implementation of MS-SSIM.
Usage:
python msssim.py --original_image=original.png --compared_image=distorted.png
"""


def _FSpecialGauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()



def MultiScaleSSIM(imgA,
                   imgB,
                   max_val=255,
                   filter_size=11,
                   filter_sigma=1.5,
                   k1=0.01,
                   k2=0.03,
                   weights=None):
    """Return the MS-SSIM score between `imgA` and `imgB`.
    This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
    Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
    similarity for image quality assessment" (2003).
    Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
    Author's MATLAB implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    Arguments:
        imgA: Numpy array holding the first RGB image batch.
        imgB: Numpy array holding the second RGB image batch.
        max_val: the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
        filter_size: Size of blur kernel to use (will be reduced for small images).
        filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
        for small images).
        k1: Constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
        k2: Constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).
        weights: List of weights for each level; if none, use five levels and the
        weights from the original paper.
    Returns:
        MS-SSIM score between `imgA` and `imgB`.
    Raises:
        RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].
    """
    if imgA.shape != imgB.shape:
        raise RuntimeError(
            'Input images must have the same shape (%s vs. %s).', imgA.shape,
            imgB.shape)

    imgA = imgA.astype(np.float32)
    imgA = imgB.astype(np.float32)
    # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
    weights = np.array(weights if weights else
                       [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    downsample_filter = np.ones((2, 2, 1)) / 4.0
    im1, im2 = [x.astype(np.float32) for x in [imgA, imgB]]
    mssim = np.array([])
    mcs = np.array([])
    for _ in range(levels):
        ssim, cs = _SSIMForMultiScale(
            im1,
            im2,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2)
        mssim = np.append(mssim, ssim)
        mcs = np.append(mcs, cs)
        filtered = [
            convolve(im, downsample_filter, mode='reflect')
            for im in [im1, im2]
        ]
        im1, im2 = [x[::2, ::2] for x in filtered]
    return (np.prod(mcs[0:levels - 1]**weights[0:levels - 1]) *
            (mssim[levels - 1]**weights[levels - 1]))

