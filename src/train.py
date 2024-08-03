import re
import time

import numpy as np

import models
from input_data import *


def main():
    with tf.device('/device:GPU:0'):
        low_res_holder = tf.placeholder(
            tf.float32,
            shape=[BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, NUM_CHENNELS])
        high_res_holder = tf.placeholder(
            tf.float32,
            shape=[BATCH_SIZE, LABEL_SIZE, LABEL_SIZE, NUM_CHENNELS])

        inferences = models.init_model(MODEL_NAME, low_res_holder)
        training_loss = models.loss(
            inferences, high_res_holder, name='training_loss', weights_decay=0)
        validation_loss = models.loss(
            inferences, high_res_holder, name='validation_loss')


    with tf.device('/device:CPU:0'):
        tf.summary.scalar('training_loss', training_loss)
        tf.summary.scalar('validation_loss', validation_loss)

        tf.summary.image('image',inferences)
        global_step = tf.Variable(0, trainable=False, name='global_step')
        # learning_rate = tf.train.piecewise_constant(
        #     global_step,
        #     [2000, 5000, 8000, 12000, 16000],
        #     [0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
        # )
        learning_rate = tf.train.inverse_time_decay(0.0001, global_step,
                                                    10000, 2)

        low_res_batch, high_res_batch = batch_queue_for_training(
            TRAINING_DATA_PATH)
        low_res_eval, high_res_eval = batch_queue_for_testing(
            VALIDATION_DATA_PATH)
    with tf.device('/device:GPU:0'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(
            training_loss, global_step=global_step)
    print('before')
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    init = (tf.global_variables_initializer(),
            tf.local_variables_initializer())
    sess.run(init)
    # Start the queue runners (make batches).
    tf.train.start_queue_runners(sess=sess)

    print('after')

    # the saver will restore all model's variables during training

    #加载上次学习点继续学习
    try:
        print('load check point now~')
        ckpt_state = tf.train.get_checkpoint_state(CHECKPOINTS_PATH)
        if not ckpt_state or not ckpt_state.model_checkpoint_path:
            print('No check point files are found!')
            pass

        ckpt_files = ckpt_state.all_model_checkpoint_paths

        num_ckpt = len(ckpt_files)
        if num_ckpt < 1:
            print('No check point files are found!')
            pass
        else:
            print('Load last check point finished!')
            saver = tf.train.Saver(
                tf.global_variables(), max_to_keep=MAX_CKPT_TO_KEEP)
            saver.restore(sess, ckpt_files[-1])
            find = re.compile('\d{3,9}')
            start_step = int(find.findall(ckpt_files[-1])[0])
            #start_step = global_step.eval()
    except:
        saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=MAX_CKPT_TO_KEEP)
        start_step = 1
    # Merge all the summaries and write them out to TRAINING_DIR
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(TRAINING_SUMMARY_PATH, sess.graph)

    print("000000000")

    for step in range(start_step, NUM_TRAINING_STEPS + 1):
        start_time = time.time()
        #print('run time:%d'%start_time)
        low_res_images, high_res_images = sess.run(
            [low_res_batch, high_res_batch])
        feed_dict = {
            low_res_holder: low_res_images,
            high_res_holder: high_res_images
        }
        _, batch_loss = sess.run(
            [train_step, training_loss], feed_dict=feed_dict)
        duration = time.time() - start_time
        assert not np.isnan(batch_loss), 'Model diverged with loss = NaN'

        if step % 100 == 0:  # show training status
            num_examples_per_step = BATCH_SIZE
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

            format_str = 'step %d, batch_loss = %.3f (%.1f examples/sec; %.3f sec/batch)'
            print(format_str % (step, batch_loss, examples_per_sec,
                                sec_per_batch))
            #saver.save(sess, join(CHECKPOINTS_PATH, 'model.ckpt'), global_step=step)

        if step % 1000 == 0:  # run validation and show its result

            low_res_images, high_res_images = sess.run(
                [low_res_eval, high_res_eval])
            feed_dict = {
                low_res_holder: low_res_images,
                high_res_holder: high_res_images
            }
            batch_loss = sess.run(validation_loss, feed_dict=feed_dict)
            print('step %d, validation loss = %.3f' % (step, batch_loss))

            summary = sess.run(merged_summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary)

        # Save the model checkpoint periodically.
        if step % 10000 == 0 or (step + 1) == NUM_TRAINING_STEPS:

            saver.save(
                sess, join(CHECKPOINTS_PATH, 'model.ckpt'), global_step=step)

    print('Training Finished!')


if __name__ == '__main__':
    main()
