from input_data import *
import models
import os
import numpy as np
import cv2 as cv
import math


def main():
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
        tf.float32, shape=[BATCH_SIZE, 120, 120, NUM_CHENNELS])
    high_res_holder = tf.placeholder(
        tf.float32, shape=[BATCH_SIZE, 120*2, 120*2, NUM_CHENNELS])

    inferences = models.init_model(MODEL_NAME, low_res_holder)
    testing_psnr = models.loss(
        inferences, high_res_holder, name='testing_psnr')

    low_res_batch, high_res_batch = batch_queue_for_testing(TESTING_DATA_PATH)

    sess = tf.Session()
    # we still need to initialize all variables even when we use Saver's restore method.
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver = tf.train.Saver(tf.global_variables())
    tf.train.start_queue_runners(sess=sess)

    best_psnr = 0
    best_ckpt = 0
    for ckpt_file in ckpt_files:
        print(ckpt_file)

        saver.restore(sess, ckpt_file)
        psnr = 0
        for i in range(NUM_TESTING_STEPS):
            low_res_images, high_res_images = sess.run(
                [low_res_batch, high_res_batch])
            feed_dict = {
                low_res_holder: low_res_images,
                high_res_holder: high_res_images
            }
            psnr += sess.run(testing_psnr, feed_dict=feed_dict)
        psnr /= NUM_TESTING_STEPS
        psnr = 20 * math.log10(255.0 / math.sqrt(psnr))
        print('Model: %s. PSNR: %.3f' % (ckpt_file, psnr))

        if psnr > best_psnr:
            best_psnr = psnr
            best_ckpt = ckpt_file


    print('Best model: %s. PSNR: %.3f' % (best_ckpt, best_psnr))

    # now, we use the best model to generate some inference patches and compare with the ground truthes
    print('\ngenerating inference patches...')
    saver.restore(sess, best_ckpt)

    for k in range(4):
        low_res_images, high_res_images = sess.run(
            [low_res_batch, high_res_batch])
        feed_dict = {
            low_res_holder: low_res_images,
            high_res_holder: high_res_images
        }
        inference_patches = sess.run(inferences, feed_dict=feed_dict)

        if not os.path.exists(INFERENCES_SAVE_PATH):
            os.mkdir(INFERENCES_SAVE_PATH)

        for i in range(BATCH_SIZE):
            low_res_input = low_res_images[i, ...]  # INPUT_SIZE x INPUT_SIZE
            ground_truth = high_res_images[i, ...]  # LABEL_SIZE x LABEL_SIZE
            inference = inference_patches[i, ...]

            crop_begin = (ground_truth.shape[0] - inference.shape[0]) // 2
            crop_end = crop_begin + inference.shape[0]
            ground_truth = ground_truth[crop_begin:crop_end, crop_begin:
                                        crop_end, ...]
            low_res_input = cv.resize(
                low_res_input, (LABEL_SIZE, LABEL_SIZE),
                interpolation=cv.INTER_CUBIC)
            low_res_input = low_res_input[crop_begin:crop_end, crop_begin:
                                          crop_end, ...]
            patch_pair = np.hstack((low_res_input, inference, ground_truth))

            # patch_pair += 0.5
            patch_pair = tf.image.convert_image_dtype(patch_pair, tf.uint8,
                                                      True)

            save_name = 'inference_%d_%d.png' % (k, i)
            cv.imwrite(
                join(INFERENCES_SAVE_PATH, save_name),
                patch_pair.eval(session=sess))

    print('Test Finished!')


if __name__ == '__main__':
    main()
