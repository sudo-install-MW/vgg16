"""
script to deal with creates the deep learning model
"""

import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()


def new_weights(shape):
    return tf.Variable(tf.compat.v1.truncated_normal(shape, stddev=0.05))


def new_biases(shape):
    return tf.compat.v1.Variable(tf.constant(0.05, shape=[shape]))


def conv_layer(input,
               filter_size=(3, 3),
               num_filters=None,
               strides=[1, 1, 1, 1],
               use_pooling=False,
               name="None"):

    filter_height = filter_size[0]
    filter_width = filter_size[1]
    out_channel = num_filters
    in_channel = input.shape[-1]

    filter_shape = (filter_height, filter_width, in_channel, out_channel)
    weights = new_weights(shape=filter_shape)
    bias = new_biases(out_channel)

    conv_layer = tf.compat.v1.nn.conv2d(input,
                                        filter=weights,
                                        strides=strides,
                                        padding='SAME') + bias

    out = tf.compat.v1.nn.relu(conv_layer)

    if use_pooling:
        out = tf.compat.v1.nn.max_pool(out,
                                       ksize=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='VALID')

    print("Layer created with shape {}".format(out.shape))

    return out


def new_fc_layer(input, num_output, name="None"):
    weights = new_weights(shape=[input.shape[-1], num_output])
    bias = new_biases(num_output)

    out = tf.matmul(input, weights) + bias

    print("FC layer")

    return tf.compat.v1.nn.relu(out)


def new_conv_to_fc_layer(input, num_output, name="None"):
    input_flatten = tf.compat.v1.layers.flatten(input)

    weights = new_weights(shape=[input_flatten.shape[-1], num_output])
    bias = new_biases(shape=num_output)

    out = tf.matmul(input_flatten, weights) + bias

    print("Created FC layer {} with shape {}".format("name", out.shape))

    return tf.compat.v1.nn.relu(out)


class VGG_16:
    def __init__(self, batch=None,
                 input_height=None,
                 input_width=None,
                 out_classes=None):

        assert type(
            out_classes) == int, "output_classes cannot be None, please pass an int value"

        self.batch = batch
        self.input_height = input_height
        self.input_width = input_width
        self.out_classes = out_classes

    def build_network_graph(self):

        self.input_layer = tf.compat.v1.placeholder(dtype=tf.float32,
                                                    shape=(self.batch,
                                                           self.input_height,
                                                           self.input_width,
                                                           3))

        conv1 = conv_layer(self.input_layer, num_filters=64)
        conv2 = conv_layer(conv1, num_filters=64, use_pooling=True)

        conv3 = conv_layer(conv2, num_filters=128)
        conv4 = conv_layer(conv3, num_filters=128, use_pooling=True)

        conv5 = conv_layer(conv4, num_filters=256)
        conv6 = conv_layer(conv5, num_filters=256)
        conv7 = conv_layer(conv6, num_filters=256, use_pooling=True)

        conv8 = conv_layer(conv7, num_filters=512)
        conv9 = conv_layer(conv8, num_filters=512)
        conv10 = conv_layer(conv9, num_filters=512, use_pooling=True)

        conv11 = conv_layer(conv10, num_filters=512)
        conv12 = conv_layer(conv11, num_filters=512)
        conv13 = conv_layer(conv12, num_filters=512, use_pooling=True)

        fc_1 = new_conv_to_fc_layer(conv13, 4096)
        fc_2 = new_fc_layer(fc_1, 4096)
        fc_3 = new_fc_layer(fc_2, 4096)

        logits = new_fc_layer(fc_3, self.out_classes)
        self.pred = tf.compat.v1.nn.softmax(logits)


if __name__ == "__main__":
    # b_image = tf.compat.v1.placeholder(
    #     dtype=tf.float32, shape=(1, 224, 224, 3))
    # layer1 = conv_layer(b_image, num_filters=16, use_pooling=True)
    # layer2 = conv_layer(layer1, num_filters=32, use_pooling=False)
    # layer3 = conv_layer(layer2, num_filters=64, use_pooling=True)
    # layer4 = conv_layer(layer3, num_filters=128, use_pooling=False)
    # logits = new_fc_layer_v2(layer4, 10)

    vgg_16 = VGG_16(input_height=224, input_width=224, out_classes=10)
    vgg_16.build_network_graph()

    print("completed")
