import numpy as np
import cv2 as cv
import os
import shutil
from configs import *


def clear_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        old_pairs = os.listdir(path)
        if len(old_pairs) > 0:
            shutil.rmtree(path)
            os.mkdir(path)


def main():
    clear_dir(TRAINING_DATA_PATH)
    clear_dir(VALIDATION_DATA_PATH)
    clear_dir(TESTING_DATA_PATH)

    assert os.path.exists(ORIGINAL_IMAGES_PATH)
    image_files = os.listdir(ORIGINAL_IMAGES_PATH)
    assert len(image_files) > 0

    # stride_pair_generate(image_files)
    # random_pair_generate(image_files)
    random_patch_generate(image_files)
    # stride_patch_generate(image_files)

    print('Data generating finished!')


def stride_patch_generate(image_files):
    file_id = 0
    for image_file in image_files:
        file_id += 1
        img = cv.imread(os.path.join(ORIGINAL_IMAGES_PATH, image_file))
        if img is None:
            continue
        print('oringinal image: ', image_file)

        rows = img.shape[0]
        cols = img.shape[1]
        sub_img_id = 0
        for i in range(0, rows - PATCH_SIZE, PATCH_GEN_STRIDE):
            for j in range(0, cols - PATCH_SIZE, PATCH_GEN_STRIDE):
                sub_img_id += 1
                sub_img_highres = img[i: i + PATCH_SIZE, j: j + PATCH_SIZE, ...]

                img_std = np.std(sub_img_highres, axis=(0, 1), dtype=np.float32)
                if np.max(img_std) < 5.0:
                    continue

                save_name = '%d_%d.png' % (file_id, sub_img_id)
                # print('saving sub image: ', save_name)
                selector = np.random.rand(1)
                if selector[0] > 0.95:
                    cv.imwrite(os.path.join(TESTING_DATA_PATH, save_name), sub_img_highres)
                elif selector[0] > 0.9:
                    cv.imwrite(os.path.join(VALIDATION_DATA_PATH, save_name), sub_img_highres)
                else:
                    cv.imwrite(os.path.join(TRAINING_DATA_PATH, save_name), sub_img_highres)


def stride_pair_generate(image_files):
    margin_size = 2
    crop_size = PATCH_SIZE + margin_size * 2  # make crop size a little larger than the patch size to revent border effects when blurring.

    downsample_size = (crop_size // 2, crop_size // 2)
    upsample_size = (crop_size, crop_size)

    downscale_methods = (cv.INTER_AREA, cv.INTER_LANCZOS4, cv.INTER_CUBIC, cv.INTER_NEAREST, cv.INTER_LINEAR)

    file_id = 0
    for image_file in image_files:
        file_id += 1
        img = cv.imread(os.path.join(ORIGINAL_IMAGES_PATH, image_file))
        if img is None:
            continue
        print('oringinal image: ', image_file)

        rows = img.shape[0]
        cols = img.shape[1]
        sub_img_id = 0
        for i in range(0, rows - crop_size, PATCH_GEN_STRIDE):
            for j in range(0, cols - crop_size, PATCH_GEN_STRIDE):
                sub_img_id += 1
                high_res_patch = img[i: i + crop_size, j: j + crop_size, ...]

                patch_std = np.std(high_res_patch, axis=(0, 1), dtype=np.float32)
                if np.max(patch_std) < 5.0:
                    continue

                r = np.random.randint(0, 3)
                if r < 5:
                    low_res_patch = cv.resize(high_res_patch, downsample_size, interpolation=downscale_methods[r])
                elif r == 5:  # Gaussian Downsample
                    low_res_patch = cv.GaussianBlur(high_res_patch, (3, 3), 0)
                    low_res_patch = cv.resize(low_res_patch, downsample_size, interpolation=cv.INTER_NEAREST)
                else:  # Box Downsample
                    low_res_patch = cv.blur(high_res_patch, (3, 3))
                    low_res_patch = cv.resize(low_res_patch, downsample_size, interpolation=cv.INTER_NEAREST)

                low_res_patch = cv.resize(low_res_patch, upsample_size, interpolation=cv.INTER_CUBIC)

                # crop to true patch size
                high_res_patch = high_res_patch[margin_size: margin_size + PATCH_SIZE, margin_size: margin_size + PATCH_SIZE, ...]
                low_res_patch = low_res_patch[margin_size: margin_size + PATCH_SIZE, margin_size: margin_size + PATCH_SIZE, ...]

                patches_pair = np.hstack((low_res_patch, high_res_patch))

                save_name = '%d_%d.png' % (file_id, sub_img_id)
                # print('saving sub image: ', save_name)
                selector = np.random.rand(1)
                if selector[0] > 0.95:
                    cv.imwrite(os.path.join(TESTING_DATA_PATH, save_name), patches_pair)
                elif selector[0] > 0.9:
                    cv.imwrite(os.path.join(VALIDATION_DATA_PATH, save_name), patches_pair)
                else:
                    cv.imwrite(os.path.join(TRAINING_DATA_PATH, save_name), patches_pair)


def random_patch_generate(image_files):
    file_id = 0
    try:
        for image_file in image_files:
            file_id += 1
            img = cv.imread(os.path.join(ORIGINAL_IMAGES_PATH, image_file))
            if img is None:
                continue
            print('oringinal image: ', image_file)

            rows = img.shape[0]
            cols = img.shape[1]
            # generate training data
            patch_id = 0
            num_patches = max(rows, cols) // PATCH_RAN_GEN_RATIO
            validation_beginning_id = num_patches - 2 * num_patches // 10
            testing_beginning_id = num_patches - num_patches // 10
            while True:
                patch_x = np.random.randint(0, cols - PATCH_SIZE)
                patch_y = np.random.randint(0, rows - PATCH_SIZE)

                high_res_patch = img[patch_y: patch_y + PATCH_SIZE, patch_x: patch_x + PATCH_SIZE, ...]
                patch_std = np.std(high_res_patch, axis=(0, 1), dtype=np.float32)
                if np.max(patch_std) < 3.0:
                    continue

                save_name = '%d_%d.png' % (file_id, patch_id)
                # print('saving sub image: ', save_name)

                if patch_id < testing_beginning_id:
                    cv.imwrite(os.path.join(TRAINING_DATA_PATH, save_name), high_res_patch)
                #elif patch_id < testing_beginning_id:
                #    cv.imwrite(os.path.join(VALIDATION_DATA_PATH, save_name), high_res_patch)
                else:
                    cv.imwrite(os.path.join(VALIDATION_DATA_PATH, save_name), high_res_patch)

                patch_id += 1
                if patch_id >= num_patches:
                    break
    except:
        pass


def random_pair_generate(image_files):
    margin_size = 2
    crop_size = PATCH_SIZE + margin_size*2  # make crop size a little larger than the patch size to revent border effects when blurring.

    downsample_size = (crop_size // 2, crop_size // 2)
    upsample_size = (crop_size, crop_size)

    downscale_methods = (cv.INTER_AREA, cv.INTER_LANCZOS4, cv.INTER_CUBIC, cv.INTER_NEAREST, cv.INTER_LINEAR)

    file_id = 0
    for image_file in image_files:
        file_id += 1
        img = cv.imread(os.path.join(ORIGINAL_IMAGES_PATH, image_file))
        if img is None:
            continue
        print('oringinal image: ', image_file)

        rows = img.shape[0]
        cols = img.shape[1]
        # generate training data
        patch_id = 0
        num_patches = max(rows, cols) // PATCH_RAN_GEN_RATIO
        validation_beginning_id = num_patches - 2 * num_patches // 10
        testing_beginning_id = num_patches - num_patches // 10
        while True:
            patch_x = np.random.randint(0, cols - crop_size)
            patch_y = np.random.randint(0, rows - crop_size)

            high_res_patch = img[patch_y: patch_y + crop_size, patch_x: patch_x + crop_size, ...]
            patch_std = np.std(high_res_patch, axis=(0, 1), dtype=np.float32)
            if np.max(patch_std) < 5.0:
                continue

            r = np.random.randint(0, 3)
            if r < 5:
                low_res_patch = cv.resize(high_res_patch, downsample_size, interpolation=downscale_methods[r])
            elif r == 5:    # Gaussian Downsample
                low_res_patch = cv.GaussianBlur(high_res_patch, (3, 3), 0)
                low_res_patch = cv.resize(low_res_patch, downsample_size, interpolation=cv.INTER_NEAREST)
            else:   # Box Downsample
                low_res_patch = cv.blur(high_res_patch, (3, 3))
                low_res_patch = cv.resize(low_res_patch, downsample_size, interpolation=cv.INTER_NEAREST)

            # r = np.random.randint(0, 2)
            # sigma = np.random.rand() * 0.4 + 0.8
            # low_res_patch = cv.GaussianBlur(low_res_patch, (2 * r + 1, 2 * r + 1), sigma)
            low_res_patch = cv.resize(low_res_patch, upsample_size, interpolation=cv.INTER_CUBIC)

            # crop to true patch size
            high_res_patch = high_res_patch[margin_size: margin_size + PATCH_SIZE, margin_size: margin_size + PATCH_SIZE, ...]
            low_res_patch = low_res_patch[margin_size: margin_size + PATCH_SIZE, margin_size: margin_size + PATCH_SIZE, ...]

            patches_pair = np.hstack((low_res_patch, high_res_patch))

            save_name = '%d_%d.png' % (file_id, patch_id)
            # print('saving sub image: ', save_name)

            if patch_id < validation_beginning_id:
                cv.imwrite(os.path.join(TRAINING_DATA_PATH, save_name), patches_pair)
            elif patch_id < testing_beginning_id:
                cv.imwrite(os.path.join(VALIDATION_DATA_PATH, save_name), patches_pair)
            else:
                cv.imwrite(os.path.join(TESTING_DATA_PATH, save_name), patches_pair)

            patch_id += 1
            if patch_id >= num_patches:
                break

if __name__ == '__main__':
    main()
