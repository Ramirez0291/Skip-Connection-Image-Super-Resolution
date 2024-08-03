import tensorflow as tf
from configs import *
from os.path import join


def rgb2yuv(rgb):
    """
    Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
    """
    rgb2yuv_filter = tf.constant([[[[0.299, -0.169,
                                     0.499], [0.587, -0.331, -0.418],
                                    [0.114, 0.499, -0.0813]]]])
    rgb2yuv_bias = tf.constant([0., 0.5, 0.5])

    temp = tf.nn.conv2d(rgb, rgb2yuv_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, rgb2yuv_bias)

    return temp


def batch_queue_for_training(data_path):
    file_names = tf.train.match_filenames_once(join(data_path, '*.png'))
    filename_queue = tf.train.string_input_producer(file_names)
    file_reader = tf.WholeFileReader()
    _, image_file = file_reader.read(filename_queue)
    patch = tf.image.decode_png(image_file, NUM_CHENNELS)
    # we must set the shape of the image before making batches
    patch.set_shape([PATCH_SIZE, PATCH_SIZE, NUM_CHENNELS])
    patch = tf.image.convert_image_dtype(patch, dtype=tf.float32)

    #test
    init = (tf.global_variables_initializer(),
            tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(file_names))

    if MAX_RANDOM_BRIGHTNESS > 0:
        patch = tf.image.random_brightness(patch, MAX_RANDOM_BRIGHTNESS)
    if len(RANDOM_CONTRAST_RANGE) == 2:
        patch = tf.image.random_contrast(patch, *RANDOM_CONTRAST_RANGE)
    patch = tf.image.random_flip_left_right(patch)
    high_res_patch = tf.image.random_flip_up_down(patch)

    crop_margin = PATCH_SIZE - LABEL_SIZE
    assert crop_margin >= 0
    if crop_margin > 1:
        high_res_patch = tf.random_crop(patch,
                                        [LABEL_SIZE, LABEL_SIZE, NUM_CHENNELS])
        # crop_pos = tf.random_uniform([2], 0, crop_margin, dtype=tf.int32)
        # offset = tf.convert_to_tensor([crop_pos[0], crop_pos[1], 0])
        # size = tf.convert_to_tensor([CROP_SIZE, CROP_SIZE, NUM_CHENNELS])
        # high_res_patch = tf.slice(patch, offset, size)
        # additional 1px shifting to low_res_patch, reducing the even/odd issue in nearest neighbor scaler.
        # shift1px = tf.random_uniform([2], -1, 2, dtype=tf.int32)
        # offset += tf.convert_to_tensor([shift1px[0], shift1px[1], 0])
        # offset = tf.clip_by_value(offset, 0, crop_margin-1)
        # low_res_patch = tf.slice(patch, offset, size)

    downscale_size = [INPUT_SIZE, INPUT_SIZE]
    resize_nn = lambda: tf.image.resize_nearest_neighbor([high_res_patch], downscale_size, True)
    resize_area = lambda: tf.image.resize_area([high_res_patch], downscale_size, True)
    resize_cubic = lambda: tf.image.resize_bicubic([high_res_patch], downscale_size, True)
    #r = tf.random_uniform([], 0, 3, dtype=tf.int32)
    #low_res_patch = tf.case({tf.equal(r, 0): resize_nn, tf.equal(r, 1): resize_area}, default=resize_cubic)[0]
    low_res_patch = resize_cubic
    #此处可以使用随机三种降采样方式，此处我用保留细节最多但有可能造成网格图像变形的BICUBIC
    # add jpeg noise to low_res_patch
    if JPEG_NOISE_LEVEL > 0:
        low_res_patch = tf.image.convert_image_dtype(
            low_res_patch, dtype=tf.uint8, saturate=True)
        jpeg_quality = 100 - 5 * JPEG_NOISE_LEVEL
        jpeg_code = tf.image.encode_jpeg(low_res_patch, quality=jpeg_quality)
        low_res_patch = tf.image.decode_jpeg(jpeg_code)
        low_res_patch = tf.image.convert_image_dtype(
            low_res_patch, dtype=tf.float32)

    # we must set tensor's shape before doing following processes
    low_res_patch.set_shape([INPUT_SIZE, INPUT_SIZE, NUM_CHENNELS])

    # add noise to low_res_patch
    if GAUSSIAN_NOISE_STD > 0:
        low_res_patch += tf.random_normal(
            low_res_patch.get_shape(), stddev=GAUSSIAN_NOISE_STD)

    low_res_patch = tf.clip_by_value(low_res_patch, 0, 1.0)
    high_res_patch = tf.clip_by_value(high_res_patch, 0, 1.0)

    #
    low_res_batch, high_res_batch = tf.train.shuffle_batch(
        [low_res_patch, high_res_patch],
        batch_size=BATCH_SIZE,
        num_threads=NUM_PROCESS_THREADS,
        capacity=MIN_QUEUE_EXAMPLES + 3 * BATCH_SIZE,
        min_after_dequeue=MIN_QUEUE_EXAMPLES)

    return low_res_batch, high_res_batch


def batch_queue_for_testing(data_path):
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(join(data_path, '*.png')))
    file_reader = tf.WholeFileReader()
    _, image_file = file_reader.read(filename_queue)
    patch = tf.image.decode_png(image_file, NUM_CHENNELS)
    # we must set the shape of the image before making batches
    patch.set_shape([240, 240, NUM_CHENNELS])
    patch = tf.image.convert_image_dtype(patch, dtype=tf.float32)

    crop_margin = 240 - 240
    offset = tf.convert_to_tensor([crop_margin // 2, crop_margin // 2, 0])
    size = tf.convert_to_tensor([240, 240, NUM_CHENNELS])
    high_res_patch = tf.slice(patch, offset, size)
    high_res_patch = patch

    downscale_size = [120, 120]
    resize_nn = lambda: tf.image.resize_nearest_neighbor([high_res_patch], downscale_size, True)
    resize_area = lambda: tf.image.resize_area([high_res_patch], downscale_size, True)
    resize_cubic = lambda: tf.image.resize_bicubic([high_res_patch], downscale_size, True)
    r = tf.random_uniform([], 0, 3, dtype=tf.int32)
    low_res_patch = tf.case({tf.equal(r, 0): resize_cubic, tf.equal(r, 1): resize_cubic}, default=resize_cubic)[0]
    #low_res_patch = resize_cubic
    #此处可以使用随机三种降采样方式，此处我用保留细节最多但有可能造成网格图像变形的BICUBIC
    # add jpeg noise to low_res_patch
    if JPEG_NOISE_LEVEL > 0:
        low_res_patch = tf.image.convert_image_dtype(
            low_res_patch, dtype=tf.uint8, saturate=True)
        jpeg_quality = 100 - 5 * JPEG_NOISE_LEVEL
        jpeg_code = tf.image.encode_jpeg(low_res_patch, quality=jpeg_quality)
        low_res_patch = tf.image.decode_jpeg(jpeg_code)
        low_res_patch = tf.image.convert_image_dtype(
            low_res_patch, dtype=tf.float32)

    # we must set tensor's shape before doing following processes
    low_res_patch.set_shape([120, 120, NUM_CHENNELS])

    # add noise to low_res_patch
    if GAUSSIAN_NOISE_STD > 0:
        low_res_patch += tf.random_normal(
            low_res_patch.get_shape(), stddev=GAUSSIAN_NOISE_STD)

    low_res_patch = tf.clip_by_value(low_res_patch, 0, 1.0)
    high_res_patch = tf.clip_by_value(high_res_patch, 0, 1.0)

    # low_res_patch -= 0.5  # approximate mean-zero data
    # high_res_patch -= 0.5
    # low_res_patch = tf.clip_by_value(low_res_patch, -0.5, 0.5)
    # high_res_patch = tf.clip_by_value(high_res_patch, -0.5, 0.5)

    # Generate batch
    low_res_batch, high_res_batch = tf.train.batch(
        [low_res_patch, high_res_patch],
        batch_size=BATCH_SIZE,
        num_threads=1,
        capacity=MIN_QUEUE_EXAMPLES + 3 * BATCH_SIZE)

    return low_res_batch, high_res_batch


def batch_queue_for_trainingY(data_path):
    file_names = tf.train.match_filenames_once(join(data_path, '*.png'))
    filename_queue = tf.train.string_input_producer(file_names)
    file_reader = tf.WholeFileReader()
    _, image_file = file_reader.read(filename_queue)
    patch = tf.image.decode_png(image_file, NUM_CHENNELS)
    patch = rgb2yuv(patch)
    # we must set the shape of the image before making batches
    patch.set_shape([PATCH_SIZE, PATCH_SIZE, NUM_CHENNELS])
    patch = tf.image.convert_image_dtype(patch, dtype=tf.float32)

    #test
    init = (tf.global_variables_initializer(),
            tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(file_names))

    if MAX_RANDOM_BRIGHTNESS > 0:
        patch = tf.image.random_brightness(patch, MAX_RANDOM_BRIGHTNESS)
    if len(RANDOM_CONTRAST_RANGE) == 2:
        patch = tf.image.random_contrast(patch, *RANDOM_CONTRAST_RANGE)
    patch = tf.image.random_flip_left_right(patch)
    high_res_patch = tf.image.random_flip_up_down(patch)

    crop_margin = PATCH_SIZE - LABEL_SIZE
    assert crop_margin >= 0
    if crop_margin > 1:
        high_res_patch = tf.random_crop(patch,
                                        [LABEL_SIZE, LABEL_SIZE, NUM_CHENNELS])
        # crop_pos = tf.random_uniform([2], 0, crop_margin, dtype=tf.int32)
        # offset = tf.convert_to_tensor([crop_pos[0], crop_pos[1], 0])
        # size = tf.convert_to_tensor([CROP_SIZE, CROP_SIZE, NUM_CHENNELS])
        # high_res_patch = tf.slice(patch, offset, size)
        # additional 1px shifting to low_res_patch, reducing the even/odd issue in nearest neighbor scaler.
        # shift1px = tf.random_uniform([2], -1, 2, dtype=tf.int32)
        # offset += tf.convert_to_tensor([shift1px[0], shift1px[1], 0])
        # offset = tf.clip_by_value(offset, 0, crop_margin-1)
        # low_res_patch = tf.slice(patch, offset, size)

    downscale_size = [INPUT_SIZE, INPUT_SIZE]
    resize_nn = lambda: tf.image.resize_nearest_neighbor([high_res_patch], downscale_size, True)
    resize_area = lambda: tf.image.resize_area([high_res_patch], downscale_size, True)
    resize_cubic = lambda: tf.image.resize_bicubic([high_res_patch], downscale_size, True)
    #r = tf.random_uniform([], 0, 3, dtype=tf.int32)
    #low_res_patch = tf.case({tf.equal(r, 0): resize_nn, tf.equal(r, 1): resize_area}, default=resize_cubic)[0]
    low_res_patch = resize_cubic
    #此处可以使用随机三种降采样方式，此处我用保留细节最多但有可能造成网格图像变形的BICUBIC
    # add jpeg noise to low_res_patch
    if JPEG_NOISE_LEVEL > 0:
        low_res_patch = tf.image.convert_image_dtype(
            low_res_patch, dtype=tf.uint8, saturate=True)
        jpeg_quality = 100 - 5 * JPEG_NOISE_LEVELresize_cubic
        jpeg_code = tf.image.encode_jpeg(low_res_patch, quality=jpeg_quality)
        low_res_patch = tf.image.decode_jpeg(jpeg_code)
        low_res_patch = tf.image.convert_image_dtype(
            low_res_patch, dtype=tf.float32)

    # we must set tensor's shape before doing following processes
    low_res_patch.set_shape([INPUT_SIZE, INPUT_SIZE, NUM_CHENNELS])

    # add noise to low_res_patch
    if GAUSSIAN_NOISE_STD > 0:
        low_res_patch += tf.random_normal(
            low_res_patch.get_shape(), stddev=GAUSSIAN_NOISE_STD)

    low_res_patch = tf.clip_by_value(low_res_patch, 0, 1.0)
    high_res_patch = tf.clip_by_value(high_res_patch, 0, 1.0)

    # low_res_patch -= 0.5    # approximate mean-zero data
    # high_res_patch -= 0.5
    # low_res_patch = tf.clip_by_value(low_res_patch, -0.5, 0.5)
    # high_res_patch = tf.clip_by_value(high_res_patch, -0.5, 0.5)

    # Generate batch
    low_res_batch, high_res_batch = tf.train.shuffle_batch(
        [low_res_patch, high_res_patch],
        batch_size=BATCH_SIZE,
        num_threads=NUM_PROCESS_THREADS,
        capacity=MIN_QUEUE_EXAMPLES + 3 * BATCH_SIZE,
        min_after_dequeue=MIN_QUEUE_EXAMPLES)

    return low_res_batch, high_res_batch


def batch_queue_for_testingY(data_path):
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(join(data_path, '*.png')))
    file_reader = tf.WholeFileReader()
    _, image_file = file_reader.read(filename_queue)
    patch = tf.image.decode_png(image_file, NUM_CHENNELS)

    # we must set the shape of the image before making batches
    patch.set_shape([PATCH_SIZE, PATCH_SIZE, NUM_CHENNELS])
    patch = tf.image.convert_image_dtype(patch, dtype=tf.float32)

    crop_margin = PATCH_SIZE - LABEL_SIZE
    offset = tf.convert_to_tensor([crop_margin // 2, crop_margin // 2, 0])
    size = tf.convert_to_tensor([LABEL_SIZE, LABEL_SIZE, NUM_CHENNELS])
    high_res_patch = tf.slice(patch, offset, size)

    downscale_size = [INPUT_SIZE_RUN, INPUT_SIZE_RUN]
    resize_nn = lambda: tf.image.resize_nearest_neighbor([high_res_patch], downscale_size, True)
    resize_area = lambda: tf.image.resize_area([high_res_patch], downscale_size, True)
    resize_cubic = lambda: tf.image.resize_bicubic([high_res_patch], downscale_size, True)
    #r = tf.random_uniform([], 0, 3, dtype=tf.int32)
    #low_res_patch = tf.case({tf.equal(r, 0): resize_nn, tf.equal(r, 1): resize_area}, default=resize_cubic)[0]
    #low_res_patch = resize_cubic
    #此处可以使用随机三种降采样方式，此处我用保留细节最多但有可能造成网格图像变形的BICUBIC
    low_res_patch = resize_cubic
    #此处可以使用随机三种降采样方式，高倍率放大需要area的方式来保护网格不变形
    # add jpeg noise to low_res_patch
    if JPEG_NOISE_LEVEL > 0:
        low_res_patch = tf.image.convert_image_dtype(
            low_res_patch, dtype=tf.uint8, saturate=True)
        jpeg_quality = 100 - 5 * JPEG_NOISE_LEVEL
        jpeg_code = tf.image.encode_jpeg(low_res_patch, quality=jpeg_quality)
        low_res_patch = tf.image.decode_jpeg(jpeg_code)
        low_res_patch = tf.image.convert_image_dtype(
            low_res_patch, dtype=tf.float32)

    # we must set tensor's shape before doing following processes
    low_res_patch.set_shape([INPUT_SIZE_RUN, INPUT_SIZE_RUN, NUM_CHENNELS])

    # add noise to low_res_patch
    if GAUSSIAN_NOISE_STD > 0:
        low_res_patch += tf.random_normal(
            low_res_patch.get_shape(), stddev=GAUSSIAN_NOISE_STD)

    low_res_patch = tf.clip_by_value(low_res_patch, 0, 1.0)
    high_res_patch = tf.clip_by_value(high_res_patch, 0, 1.0)

    # low_res_patch -= 0.5  # approximate mean-zero data
    # high_res_patch -= 0.5
    # low_res_patch = tf.clip_by_value(low_res_patch, -0.5, 0.5)
    # high_res_patch = tf.clip_by_value(high_res_patch, -0.5, 0.5)

    # Generate batch
    low_res_batch, high_res_batch = tf.train.batch(
        [low_res_patch, high_res_patch],
        batch_size=BATCH_SIZE,
        num_threads=1,
        capacity=MIN_QUEUE_EXAMPLES + 3 * BATCH_SIZE)

    return low_res_batch, high_res_batch