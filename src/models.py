from configs import *
from layers import *
from SSIMtest import *


def init_model(name, patches):
    if name == 'vgg7':
        return vgg7(patches)
    elif name == 'vgg7woc':
        return vgg7withoutdeconv2d(patches)
    elif name == 'RESCNN':
        return RESCNN(patches)
    elif name == '1135':
        return mod1135(patches)
    elif name == 'modplus':
        return modplus(patches)
    elif name == 'mod2plus':
        return mod2plus(patches)
    elif name == 'mod2plusY':
        return mod2plusY(patches)
    elif name == 'hrscsr':
        return hrscsr(patches)


def mod1135(patches, name='1135'):
    with tf.variable_scope(name):
        #出现35像素为单位的棋盘效应

        upscaled_patches = tf.image.resize_bicubic(
            patches, [INPUT_SIZE, INPUT_SIZE], True)
        #对低分辨率图像进行bicubic插值，提升到高分辨率(伪)
        conv1 = conv2d(patches, 11, 11, 128, padding='VALID', name='conv1')
        lrelu1 = leaky_relu(conv1, name='leaky_relu1')
        conv2 = conv2d(lrelu1, 3, 3, 64, padding='VALID', name='conv2')
        lrelu2 = leaky_relu(conv2, name='leaky_relu2')
        conv3 = conv2d(
            lrelu2, 5, 5, NUM_CHENNELS, padding='VALID', name='conv3')
        lrelu3 = leaky_relu(conv3, name='leaky_relu3')
        batch_size = int(lrelu3.get_shape()[0])
        rows = int(lrelu3.get_shape()[1])
        cols = int(lrelu3.get_shape()[2])
        channels = int(patches.get_shape()[3])
        return deconv2d(
            lrelu3,
            4,
            4, [batch_size, rows * 2, cols * 2, channels],
            stride=(2, 2),
            name='deconv_out')


def RESCNN(patches, name='RESCNN'):
    with tf.variable_scope(name):
        conv1 = conv2d(patches, 3, 3, 32, padding='VALID', name='conv1')
        lrelu1 = leaky_relu(conv1, name='leaky_relu1')
        #此处将部分值传到两层之后
        conv2 = conv2d(lrelu1, 1, 1, 64, padding='VALID', name='conv2')
        lrelu2 = leaky_relu(conv2, name='leaky_relu2')
        conv3_a = conv2d(lrelu2, 3, 3, 64, padding='VALID', name='conv3_a')
        lrelu3_a = leaky_relu(conv3_a, name='leaky_relu3_a')

        conv3_b = conv2d(lrelu1, 3, 3, 64, padding='VALID', name='conv3_b')
        lrelu3_b = leaky_relu(conv3_b, name='leaky_relu3_b')
        #lrelu3 = lrelu3_a+lrelu3_b
        lrelu3 = tf.add(lrelu3_a, lrelu3_b)
        #lrelu3 = batch_norm(lrelu3)

        #受到传得结果，合并成lrelu3，继续传到两层后
        conv4 = conv2d(lrelu3, 1, 1, 128, padding='VALID', name='conv4')
        lrelu4 = leaky_relu(conv4, name='leaky_relu3')
        conv5_a = conv2d(lrelu4, 3, 3, 128, padding='VALID', name='conv5_a')
        lrelu5_a = leaky_relu(conv5_a, name='leaky_relu5_a')

        conv5_b = conv2d(lrelu3, 3, 3, 128, padding='VALID', name='conv5_b')
        lrelu5_b = leaky_relu(conv5_b, name='leaky_relu5_b')
        #lrelu5=lrelu5_a+lrelu5_b
        lrelu5 = tf.add(lrelu5_a, lrelu5_b)
        #lrelu5=batch_norm(lrelu5)

        conv6 = conv2d(lrelu5, 3, 3, 256, padding='VALID', name='conv6')
        lrelu6 = leaky_relu(conv6, name='leaky_relu6')
        batch_size = int(lrelu6.get_shape()[0])
        rows = int(lrelu6.get_shape()[1])
        cols = int(lrelu6.get_shape()[2])
        channels = int(patches.get_shape()[3])
        # 最后通过反卷积层的放大，将图片真正的尺寸X2化
        return deconv2d(
            lrelu6,
            4,
            4, [batch_size, rows * 2, cols * 2, channels],
            stride=(2, 2),
            name='deconv_out')


def mod935(patches, name='935'):
    with tf.variable_scope(name):
        upscaled_patches = tf.image.resize_bicubic(
            patches, [INPUT_SIZE, INPUT_SIZE], True)
        #对低分辨率图像进行bicubic插值，提升到高分辨率(伪)
        conv1 = conv2d(
            upscaled_patches, 9, 9, 64, padding='VALID', name='conv1')
        lrelu1 = leaky_relu(conv1, name='leaky_relu1')
        conv2 = conv2d(lrelu1, 3, 3, 32, padding='VALID', name='conv2')
        lrelu2 = leaky_relu(conv2, name='leaky_relu2')
        conv3 = conv2d(
            lrelu2, 5, 5, NUM_CHENNELS, padding='VALID', name='conv3')

        return conv3


def vgg7withoutdeconv2d(patches, name='vgg7woc'):
    with tf.variable_scope(name):
        # conv2d二维卷积层，输入前一层，边界处理VALID
        conv1 = conv2d(patches, 3, 3, 16, padding='VALID', name='conv1')
        lrelu1 = leaky_relu(conv1, name='leaky_relu1')
        conv2 = conv2d(lrelu1, 3, 3, 32, padding='VALID', name='conv2')
        lrelu2 = leaky_relu(conv2, name='leaky_relu2')
        conv3 = conv2d(lrelu2, 3, 3, 64, padding='VALID', name='conv3')
        lrelu3 = leaky_relu(conv3, name='leaky_relu3')
        conv4 = conv2d(lrelu3, 3, 3, 128, padding='VALID', name='conv4')
        lrelu4 = leaky_relu(conv4, name='leaky_relu4')
        conv5 = conv2d(lrelu4, 3, 3, 128, padding='VALID', name='conv5')
        lrelu5 = leaky_relu(conv5, name='leaky_relu5')
        conv6 = conv2d(lrelu5, 3, 3, 256, padding='VALID', name='conv6')
        lrelu6 = leaky_relu(conv6, name='leaky_relu6')
        return conv2d(
            lrelu6, 3, 3, NUM_CHENNELS, padding='VALID', name='conv_out')


def vgg7(patches, name='vgg7'):
    with tf.variable_scope(name):
        conv1 = conv2d(patches, 3, 3, 16, padding='VALID', name='conv1')
        lrelu1 = leaky_relu(conv1, name='leaky_relu1')
        conv2 = conv2d(lrelu1, 3, 3, 32, padding='VALID', name='conv2')
        lrelu2 = leaky_relu(conv2, name='leaky_relu2')
        conv3 = conv2d(lrelu2, 3, 3, 64, padding='VALID', name='conv3')
        lrelu3 = leaky_relu(conv3, name='leaky_relu3')
        conv4 = conv2d(lrelu3, 3, 3, 128, padding='VALID', name='conv4')
        lrelu4 = leaky_relu(conv4, name='leaky_relu4')
        conv5 = conv2d(lrelu4, 3, 3, 128, padding='VALID', name='conv5')
        lrelu5 = leaky_relu(conv5, name='leaky_relu5')
        conv6 = conv2d(lrelu5, 3, 3, 256, padding='VALID', name='conv6')
        lrelu6 = leaky_relu(conv6, name='leaky_relu6')

        batch_size = int(lrelu6.get_shape()[0])
        rows = int(lrelu6.get_shape()[1])
        cols = int(lrelu6.get_shape()[2])
        channels = int(patches.get_shape()[3])
        # 最后通过反卷积层的放大，将图片真正的尺寸X2化
        return deconv2d(
            lrelu6,
            4,
            4, [batch_size, rows * 2, cols * 2, channels],
            stride=(2, 2),
            name='deconv_out')


def modplus(patches, name='modplus'):
    with tf.variable_scope(name):
        conv0 = conv2dplus(patches, 3, 3, 56, padding='VALID', name='conv0')
        prelu0 = para_relu(conv0, name='prelu0')
        conv1 = conv2dplus(prelu0, 3, 3, 56, padding='VALID', name='conv1')
        prelu1 = para_relu(conv1, name='prelu1')
        conv2 = conv2dplus(prelu1, 1, 1, 12, padding='VALID', name='conv2')
        prelu2 = para_relu(conv2, name='prelu2')
        conv3_1 = conv2dplus(
            prelu2,
            3,
            3,
            12,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='conv3_1')
        prelu3_1 = para_relu(conv3_1, name='prelu3_1')
        conv3_2 = conv2dplus(
            prelu3_1,
            3,
            3,
            12,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='conv3_2')
        prelu3_2 = para_relu(conv3_2, name='prelu3_2')
        conv3_3 = conv2dplus(
            prelu3_2,
            3,
            3,
            12,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='conv3_3')
        prelu3_3 = para_relu(conv3_3, name='prelu3_3')
        conv3_4 = conv2dplus(
            prelu3_3,
            3,
            3,
            12,
            pad=1,
            padding='SAME',
            gaussian=0.1897,
            name='conv3_4')
        prelu3_4 = para_relu(conv3_4, name='prelu3_4')
        conv4 = conv2dplus(prelu3_4, 1, 1, 56, padding='VALID', name='conv4')
        prelu4 = para_relu(conv4, name='prelu4')

        batch_size = int(prelu4.get_shape()[0])
        rows = int(prelu4.get_shape()[1])
        cols = int(prelu4.get_shape()[2])
        channels = int(patches.get_shape()[3])
        return deconv2dplus(
            prelu4,
            9,
            9, [batch_size, rows * 3, cols * 3, channels],
            stride=(3, 3),
            pad=4,
            padding='SAME',
            name='deconv_out')


def mod2plus(patches, name='modplus'):
    with tf.variable_scope(name):
        conv0 = conv2dplus(patches, 3, 3, 64, padding='VALID', name='conv0')
        prelu0 = para_relu(conv0, name='prelu0')
        conv1 = conv2dplus(prelu0, 3, 3, 128, padding='VALID', name='conv1')
        prelu1 = para_relu(conv1, name='prelu1')
        conv2 = conv2dplus(prelu1, 1, 1, 64, padding='VALID', name='conv2')
        prelu2 = para_relu(conv2, name='prelu2')
        conv3_1 = conv2dplus(
            prelu2,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='conv3_1')
        prelu3_1 = para_relu(conv3_1, name='prelu3_1')
        conv3_2 = conv2dplus(
            prelu3_1,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='conv3_2')
        prelu3_2 = para_relu(conv3_2, name='prelu3_2')
        conv3_3 = conv2dplus(
            prelu3_2,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='conv3_3')
        prelu3_3 = para_relu(conv3_3, name='prelu3_3')
        conv3_4 = conv2dplus(
            prelu3_3,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1897,
            name='conv3_4')
        prelu3_4 = para_relu(conv3_4, name='prelu3_4')
        conv4 = conv2dplus(prelu3_4, 1, 1, 128, padding='VALID', name='conv4')
        prelu4 = para_relu(conv4, name='prelu4')
        maxlayer = prelu4 + prelu1

        batch_size = int(maxlayer.get_shape()[0])
        rows = int(maxlayer.get_shape()[1])
        cols = int(maxlayer.get_shape()[2])
        channels = int(patches.get_shape()[3])
        return deconv2dplus(
            maxlayer,
            9,
            9,
            [batch_size, rows * SCALE_FACTOR, cols * SCALE_FACTOR, channels],
            stride=(SCALE_FACTOR, SCALE_FACTOR),
            pad=4,
            padding='SAME',
            name='deconv_out')


def mod2plusY(patches, name='modplus'):
    with tf.variable_scope(name):
        # layer1
        conv1 = conv2dplus(patches, 3, 3, 64, padding='VALID', name='conv1')
        prelu1 = para_relu(conv1, name='prelu1')
        #add test
        conv1_1 = conv2dplus(
            prelu1, 3, 3, 128, padding='VALID', name='conv1_1')
        prelu1_1 = para_relu(conv1_1, name='prelu1_1')
        #layer2
        conv2 = conv2dplus(prelu1_1, 3, 3, 128, padding='VALID', name='conv2')
        prelu2 = para_relu(conv2, name='prelu2')
        #layer3
        conv3 = conv2dplus(prelu2, 1, 1, 64, padding='VALID', name='conv3')
        prelu3 = para_relu(conv3, name='prelu3')
        #res block 1

        #layer 4_1
        skipblock1 = skipblock(
            prelu3,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='skipblock1')
       
        #layer4_2
        skipblock2 = skipblock(
            skipblock1,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='skipblock2')

        #layer4_3
        skipblock3 = skipblock(
            skipblock1 + skipblock2,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='skipblock3')

        #layer4_4
        skipblock4 = skipblock(
            skipblock1 + skipblock2 + skipblock3,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1897,
            name='skipblock4')

        #layer4_5
        skipblock5 = skipblock(
            skipblock1 + skipblock2 + skipblock3 + skipblock4,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1897,
            name='skipblock5')

        conv5 = conv2dplus(
            skipblock5, 1, 1, 128, padding='VALID', name='conv5')
        prelu5 = para_relu(conv5, name='prelu5')
        maxlayer = 0.1 * prelu2 + prelu5

        batch_size = int(maxlayer.get_shape()[0])
        rows = int(maxlayer.get_shape()[1])
        cols = int(maxlayer.get_shape()[2])
        channels = int(patches.get_shape()[3])
        deconv = deconv2dplus(
            maxlayer,
            4,
            4, [batch_size, rows * 2, cols * 2, 32],
            stride=(2, 2),
            pad=4,
            padding='SAME',
            name='deconv_out')
        output = conv2dplus(deconv,1, 1, channels,padding='VALID', name='output')
        return output
        


#high_rage_skip_connect_super_resolution network_
def hrscsr(patches, name='hrscsr'):
    with tf.variable_scope(name):
        # layer1
        conv1 = conv2dplus(patches, 3, 3, 64, padding='VALID', name='conv1')
        prelu1 = para_relu(conv1, name='prelu1')
        #add test
        conv1_1 = conv2dplus(
            prelu1, 3, 3, 128, padding='VALID', name='conv1_1')
        prelu1_1 = para_relu(conv1_1, name='prelu1_1')
        #layer2
        conv2 = conv2dplus(prelu1_1, 3, 3, 128, padding='VALID', name='conv2')
        prelu2 = para_relu(conv2, name='prelu2')
        #layer3
        conv3 = conv2dplus(prelu2, 1, 1, 64, padding='VALID', name='conv3')
        prelu3 = para_relu(conv3, name='prelu3')

        #skip_connect block 1-----------------------------------------------
        #block 1_layer 1
        block1_conv1 = conv2dplus(
            prelu3,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='block1_conv1')
        block1_prelu1 = para_relu(block1_conv1, name='block1_prelu1')
        block1_batch_norm1 = batch_norm(
            block1_prelu1, name='block1_batch_norm1')
        #block 1_layer 2
        block1_conv2 = conv2dplus(
            block1_batch_norm1,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='block1_conv2')
        block1_prelu2 = para_relu(block1_conv2, name='block1_prelu2')
        block1_batch_norm2 = batch_norm(
            block1_prelu2, name='block1_batch_norm2')
        #block 1_layer 3
        block1_conv3 = conv2dplus(
            block1_batch_norm1 + block1_batch_norm2,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='block1_conv3')
        block1_prelu3 = para_relu(block1_conv3, name='block1_prelu3')
        block1_batch_norm3 = batch_norm(
            block1_prelu3, name='block1_batch_norm3')
        #block 1_layer 4
        block1_conv4 = conv2dplus(
            block1_batch_norm1 + block1_batch_norm2 + block1_batch_norm3,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1897,
            name='block1_conv4')
        block1_prelu4 = para_relu(block1_conv4, name='block1_prelu4')
        block1_batch_norm4 = batch_norm(
            block1_prelu4, name='block1_batch_norm4')
        #block 1_layer 5
        block1_conv5 = conv2dplus(
            block1_batch_norm1 + block1_batch_norm2 + block1_batch_norm3 +
            block1_batch_norm4,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1897,
            name='block1_conv5')
        block1_prelu5 = para_relu(block1_conv5, name='block1_prelu5')
        block1_batch_norm5 = batch_norm(
            block1_prelu5, name='block1_batch_norm5')

        block2 = block1_batch_norm5 + prelu3
        #skip_connect block 2----------------------------------------------
        #block 2_layer 1
        block2_conv1 = conv2dplus(
            block2,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='block2_conv1')
        block2_prelu1 = para_relu(block2_conv1, name='block2_prelu1')
        block2_batch_norm1 = batch_norm(
            block2_prelu1, name='block2_batch_norm1')
        #block 2_layer 2
        block2_conv2 = conv2dplus(
            block2_batch_norm1,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='block2_conv2')
        block2_prelu2 = para_relu(block2_conv2, name='block2_prelu2')
        block2_batch_norm2 = batch_norm(
            block2_prelu2, name='block2_batch_norm2')
        #block 2_layer 3
        block2_conv3 = conv2dplus(
            block2_batch_norm1 + block2_batch_norm2,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='block2_conv3')
        block2_prelu3 = para_relu(block2_conv3, name='block2_prelu3')
        block2_batch_norm3 = batch_norm(
            block2_prelu3, name='block2_batch_norm3')
        #block 2_layer 4
        block2_conv4 = conv2dplus(
            block2_batch_norm1 + block2_batch_norm2 + block2_batch_norm3,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1897,
            name='block2_conv4')
        block2_prelu4 = para_relu(block2_conv4, name='block2_prelu4')
        block2_batch_norm4 = batch_norm(
            block2_prelu4, name='block2_batch_norm4')
        #block 2_layer 5
        block2_conv5 = conv2dplus(
            block2_batch_norm1 + block2_batch_norm2 + block2_batch_norm3 +
            block2_batch_norm4,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1897,
            name='block2_conv5')
        block2_prelu5 = para_relu(block2_conv5, name='block2_prelu5')
        block2_batch_norm5 = batch_norm(
            block2_prelu5, name='block2_batch_norm5')

        block3 = block2_batch_norm5 + prelu3
        #skip_connect block 3-----------------------------------------------
        #block 3_layer 1
        block3_conv1 = conv2dplus(
            block3,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='block3_conv1')
        block3_prelu1 = para_relu(block3_conv1, name='block3_prelu1')
        block3_batch_norm1 = batch_norm(
            block3_prelu1, name='block3_batch_norm1')
        #block 3_layer 2
        block3_conv2 = conv2dplus(
            block3_batch_norm1,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='block3_conv2')
        block3_prelu2 = para_relu(block3_conv2, name='block3_prelu2')
        block3_batch_norm2 = batch_norm(
            block3_prelu2, name='block3_batch_norm2')
        #block 3_layer 3
        block3_conv3 = conv2dplus(
            block3_batch_norm1 + block3_batch_norm2,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='block3_conv3')
        block3_prelu3 = para_relu(block3_conv3, name='block3_prelu3')
        block3_batch_norm3 = batch_norm(
            block3_prelu3, name='block3_batch_norm3')
        #block 3_layer 4
        block3_conv4 = conv2dplus(
            block3_batch_norm1 + block3_batch_norm2 + block3_batch_norm3,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1897,
            name='block3_conv4')
        block3_prelu4 = para_relu(block3_conv4, name='block3_prelu4')
        block3_batch_norm4 = batch_norm(
            block3_prelu4, name='block3_batch_norm4')
        #block 3_layer 5
        block3_conv5 = conv2dplus(
            block3_batch_norm1 + block3_batch_norm2 + block3_batch_norm3 +
            block3_batch_norm4,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1897,
            name='block3_conv5')
        block3_prelu5 = para_relu(block3_conv5, name='block3_prelu5')
        block3_batch_norm5 = batch_norm(
            block3_prelu5, name='block3_batch_norm5')

        block4 = block3_batch_norm5 + prelu3
        #skip_connect block 4---------------------------------------------------------------
        #block 4_layer 1
        block4_conv1 = conv2dplus(
            block4,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='block4_conv1')
        block4_prelu1 = para_relu(block4_conv1, name='block4_prelu1')
        block4_batch_norm1 = batch_norm(
            block4_prelu1, name='block4_batch_norm1')
        #block 4_layer 2
        block4_conv2 = conv2dplus(
            block4_batch_norm1,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='block4_conv2')
        block4_prelu2 = para_relu(block4_conv2, name='block4_prelu2')
        block4_batch_norm2 = batch_norm(
            block4_prelu2, name='block4_batch_norm2')
        #block 4_layer 3
        block4_conv3 = conv2dplus(
            block4_batch_norm1 + block4_batch_norm2,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='block4_conv3')
        block4_prelu3 = para_relu(block4_conv3, name='block4_prelu3')
        block4_batch_norm3 = batch_norm(
            block4_prelu3, name='block4_batch_norm3')
        #block 4_layer 4
        block4_conv4 = conv2dplus(
            block4_batch_norm1 + block4_batch_norm2 + block4_batch_norm3,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1897,
            name='block4_conv4')
        block4_prelu4 = para_relu(block4_conv4, name='block4_prelu4')
        block4_batch_norm4 = batch_norm(
            block4_prelu4, name='block4_batch_norm4')
        #block 4_layer 5
        block4_conv5 = conv2dplus(
            block4_batch_norm1 + block4_batch_norm2 + block4_batch_norm3 +
            block4_batch_norm4,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1897,
            name='block4_conv5')
        block4_prelu5 = para_relu(block4_conv5, name='block4_prelu5')
        block4_batch_norm5 = batch_norm(
            block4_prelu5, name='block4_batch_norm5')
        #-----------------------------------------------------------------------------
        #blockout = block4_batch_norm5 + block3_batch_norm5 + block2_batch_norm5 + block1_batch_norm5 + prelu3

        conv5 = conv2dplus(block4_batch_norm5, 1, 1, 128, padding='VALID', name='conv5')
        prelu5 = para_relu(conv5, name='prelu5')
        maxlayer = 0.1*prelu2 + prelu5

        batch_size = int(maxlayer.get_shape()[0])
        rows = int(maxlayer.get_shape()[1])
        cols = int(maxlayer.get_shape()[2])
        channels = int(patches.get_shape()[3])
        return deconv2dplus(
            maxlayer,
            4,
            4, [batch_size, rows * 2, cols * 2, channels],
            stride=(2, 2),
            pad=4,
            padding='SAME',
            name='deconv_out')


def mod3plus(patches, name='modplus'):
    with tf.variable_scope(name):
        conv1 = conv2dplus(patches, 3, 3, 32, padding='VALID', name='conv1')
        prelu1 = para_relu(conv1, name='prelu1')
        conv2 = conv2dplus(prelu1, 3, 3, 64, padding='VALID', name='conv2')
        prelu2 = para_relu(conv2, name='prelu2')
        conv3 = conv2dplus(prelu2, 3, 3, 64, padding='VALID', name='conv3')
        prelu3 = para_relu(conv3, name='prelu3')
        conv4 = conv2dplus(prelu3, 3, 3, 128, padding='VALID', name='conv4')
        prelu4 = para_relu(conv4, name='prelu4')
        conv5 = conv2dplus(prelu4, 3, 3, 128, padding='VALID', name='conv5')
        prelu5 = para_relu(conv5, name='prelu5')
        conv6 = conv2dplus(prelu5, 3, 3, 256, padding='VALID', name='conv6')
        prelu6 = para_relu(conv6, name='prelu6')
        conv7 = conv2dplus(prelu6, 3, 3, 128, padding='VALID', name='conv7')
        prelu7 = para_relu(conv7, name='prelu7')
        convnon_map_1 = conv2dplus(
            prelu7,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='convnon_map_1')
        prelu3_1 = para_relu(convnon_map_1, name='prelu3_1')
        convnon_map_2 = conv2dplus(
            prelu3_1,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='convnon_map_2')
        prelu3_2 = para_relu(convnon_map_2, name='prelu3_2')
        convnon_map_3 = conv2dplus(
            prelu3_2,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='convnon_map_3')
        prelu3_3 = para_relu(convnon_map_3, name='prelu3_3')
        convnon_map_4 = conv2dplus(
            prelu3_3,
            3,
            3,
            64,
            pad=1,
            padding='SAME',
            gaussian=0.1897,
            name='convnon_map_4')
        prelu3_4 = para_relu(convnon_map_4, name='prelu3_4')
        conv8 = conv2dplus(prelu3_4, 1, 1, 128, padding='VALID', name='conv8')
        prelu8 = para_relu(conv8, name='prelu8')
        maxlayer = prelu8 + prelu7

        batch_size = int(maxlayer.get_shape()[0])
        rows = int(maxlayer.get_shape()[1])
        cols = int(maxlayer.get_shape()[2])
        channels = int(patches.get_shape()[3])
        return deconv2dplus(
            maxlayer,
            SCALE_FACTOR**2,
            SCALE_FACTOR**2,
            [batch_size, rows * SCALE_FACTOR, cols * SCALE_FACTOR, channels],
            stride=(SCALE_FACTOR, SCALE_FACTOR),
            pad=4,
            padding='SAME',
            name='deconv_out')


def loss(inferences,
         ground_truthes,
         huber_width=0.1,
         weights_decay=0,
         name='loss'):
    with tf.name_scope(name):
        slice_begin = (
            int(ground_truthes.get_shape()[1]) - int(inferences.get_shape()[1])
        ) // 2
        slice_end = int(inferences.get_shape()[1]) + slice_begin
        delta = inferences - ground_truthes[:, slice_begin:slice_end,
                                            slice_begin:slice_end, :]

        delta *= [[[[0.11448, 0.58661, 0.29891]]]]  # weights of B, G and R
        #l2_loss = tf.nn.l2_loss(delta)
        # 这是l2_loss的计算，文献中是l2_loss**2的平均
        l2_loss = tf.pow(delta, 2)

        #将l2loss的矩阵从1->2->3轴来压缩后取平均
        mse_loss = tf.reduce_mean(tf.reduce_sum(l2_loss, axis=[1, 2, 3]))
        #以上，得到MSE
        if weights_decay > 0:
            weights = tf.get_collection('weights')
            reg_loss = weights_decay * tf.reduce_mean(
                tf.reduce_sum(
                    tf.stack([tf.nn.l2_loss(i) for i in weights]),
                    name='regularization_loss'))
            return mse_loss + reg_loss
        else:
            return mse_loss


def psnr(inferences,ground_truthes,name='psnr'):
    with tf.name_scope(name):
        psnr = tf.image.psnr(inferences,ground_truthes,max_val=1.0)
        return psnr
