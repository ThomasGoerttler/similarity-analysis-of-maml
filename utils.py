""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

import matplotlib.pyplot as plt

DISTINCT_COLORS = ["#e6194b", "#3cb44b", "#0082c8", "#f58230",
                       "#911eb4", "#46f0f0", "#f032e6", "#d2f53c", "#ffe119"]


FLAGS = flags.FLAGS

## Image helper
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
        for i, path in zip(labels, paths) \
        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images

## Network helpers
def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID', residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    if FLAGS.max_pool:
        conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    else:
        conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
    normed = normalize(conv_output, activation, reuse, scope)
    if FLAGS.max_pool:
        normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
    return normed

def normalize(inp, activation, reuse, scope):
    if FLAGS.norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size


def load_model(dir, exp_string, saver, sess, iteration = -1):
    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(dir + '' + exp_string)
        print(iteration, dir + '' + exp_string)
        if iteration > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(iteration)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

def interpret_steps(string):
    if string.startswith("range"):
        string = string.replace("range(", "")
        string = string.replace(")", "")
        numbers = string.split(",")
        print(numbers)
        return list(range(int(numbers[0]), int(numbers[1]), int(numbers[2])))
    elif string.startswith("["):
        string = string.replace("[", "")
        string = string.replace("]", "")
        return string.split(",")
    else:
        try:
            return [int(string)]
        except:
            return 0


def plot_neighbour_analysis(steps, final_base_representation, layer_names):


    from rsa import rsa
    similaritiy_to_pre = [[] for _ in range(len(final_base_representation))]
    for i in range(len(final_base_representation)):
        for j in range(1, len(final_base_representation[i])):
            similaritiy_to_pre[i].append(rsa(final_base_representation[i][[j - 1, j], :]))


    fig = plt.figure(figsize=(8, 4))
    plt.title("Dissimilarity of meta-initilizations w.r.t representation 1000 trainingsteps before")
    plt.xlabel("Number of training steps")
    plt.ylabel("Dissimilarity")
    plt.yscale('symlog', linthreshy=0.015)
    print(np.array(steps)[1:])
    for i in range(5):
        plt.plot(np.array(steps)[1:], similaritiy_to_pre[i], label=layer_names[i])
        print(similaritiy_to_pre[i])
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.show()


def plot_base_analysis(steps, final_mean_diff_to_base, final_std_diff_to_base, layer_names):

    fig = plt.figure(figsize=(8, 4))
    plt.title("Mean dissimilarity w.r.t base presentation")
    plt.xlabel("Number of training steps")
    plt.ylabel("Dissimilarity")
    for i in range(5):
        plt.errorbar(np.array(steps), final_mean_diff_to_base[i], final_std_diff_to_base[i], label=layer_names[i])
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.show()

def plot_performance(steps, accs, NUM_TEST_POINTS):

    NUM_SAMPLE = 4
    acc_singles = [[] for _ in range(NUM_SAMPLE)]
    acc_sum = []
    for acc in accs:
        for i in range(NUM_SAMPLE):
            acc_singles[i].append(acc[i + 1])  # First is evaluation task
        acc_sum.append(np.mean(acc))


    fig = plt.figure(figsize=(8, 4))

    plt.title("Performance of four selected tasks")
    for i in range(NUM_SAMPLE):
        plt.plot(steps, acc_singles[i], color=DISTINCT_COLORS[i], label=f"Task {i+1}")

    plt.plot(steps, acc_sum, color="black", label=f"Average over {NUM_TEST_POINTS} tasks")
    plt.xlabel("Number of training steps")
    plt.ylabel("Accuracy after 5 inner gradient steps")
    plt.legend(loc="lower right")

    plt.show()
