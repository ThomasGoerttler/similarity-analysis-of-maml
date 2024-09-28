from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers

from rsa import *
from cka import *

import matplotlib.pyplot as plt

from utils import apply_representation_similarity

# heiner activation maximization filters early layers

# based on https://github.com/zonghua94/mnist/blob/master/mnist_cnn.py

def compute_accuracy(v_x, v_y):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x: v_x})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={x: v_x, y: v_y})
    return result

def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID', residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]
    conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
    normed = tf_layers.batch_norm(conv_output, activation_fn=activation, reuse=reuse, scope=scope)
    return normed

def reshape_elems_of_list(layers, shape = (10000, -1)):
    reshaped_layers = []
    for layer in layers:
        layer = np.reshape(layer, shape)
        reshaped_layers.append(layer)
    return reshaped_layers

# load mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
# reshape(data you want to reshape, [-1, reshape_height, reshape_weight, imagine layers]) image layers=1 when the imagine is in white and black, =3 when the imagine is RGB
x_image = tf.reshape(x, [-1, 28, 28, 1])

weights = {}
convolution = True
if convolution:

    dtype = tf.float32
    conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)

    weights['conv1'] = tf.get_variable('conv1', [3, 3, 1, 64], initializer=conv_initializer, dtype=dtype)
    weights['b1'] = tf.Variable(tf.zeros([64]))
    weights['conv2'] = tf.get_variable('conv2', [3, 3, 64, 64], initializer=conv_initializer, dtype=dtype)
    weights['b2'] = tf.Variable(tf.zeros([64]))
    weights['conv3'] = tf.get_variable('conv3', [3, 3, 64, 64], initializer=conv_initializer, dtype=dtype)
    weights['b3'] = tf.Variable(tf.zeros([64]))
    weights['conv4'] = tf.get_variable('conv4', [3, 3, 64, 64], initializer=conv_initializer, dtype=dtype)
    weights['b4'] = tf.Variable(tf.zeros([64]))

    weights['w5'] = tf.Variable(tf.random_normal([64, 10]), name='w5')
    weights['b5'] = tf.Variable(tf.zeros([10]), name='b5')

    tvars = tf.trainable_variables()

    scope = ""
    hidden1 = conv_block(x_image, weights['conv1'], weights['b1'], False, scope + '0')
    hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], False, scope + '1')
    hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], False, scope + '2')
    hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], False, scope + '3')

    hidden4 = tf.reduce_mean(hidden4, [1, 2])
    out = tf.matmul(hidden4, weights['w5']) + weights['b5']
    prediction = tf.nn.softmax(out)

    tvars = tf.trainable_variables()

    layer_names = ["Pooling layer 1", "Pooling layer 2", "Pooling layer 3", "Pooling layer 4", "Logits/Head"]
    layer_names = ["pool 1", "pool 2", "pool 3", "pool 4", "logits"]
else:

    weights = {}

    dims = [200, 100, 50, 20]

    weights['w1'] = tf.Variable(tf.truncated_normal([784, dims[0]], stddev=0.01))
    weights['b1'] = tf.Variable(tf.zeros(dims[0]))

    for i, dim in enumerate(dims):
        if i == len(dims) -1:
            break
        weights['w'+str(i+2)] = tf.Variable(tf.truncated_normal([dims[i], dims[i+1]], stddev=0.01))
        weights['b'+str(i+2)] = tf.Variable(tf.zeros(dims[i+1]))

    weights['w5'] = tf.Variable(tf.random_normal([dims[-1], 10]), name='w5')
    weights['b5'] = tf.Variable(tf.zeros([10]), name='b5')

    x_image = tf.reshape(x_image, [-1, 784])

    hidden1 = tf.nn.relu(tf_layers.batch_norm(tf.matmul(x_image, weights['w1']) + weights['b1']))
    hidden2 = tf.nn.relu(tf_layers.batch_norm(tf.matmul(hidden1, weights['w2']) + weights['b2']))
    hidden3 = tf.nn.relu(tf_layers.batch_norm(tf.matmul(hidden2, weights['w3']) + weights['b3']))
    hidden4 = tf.nn.relu(tf_layers.batch_norm(tf.matmul(hidden3, weights['w4']) + weights['b4']))
    out = tf.matmul(hidden4, weights['w5']) + weights['b5']
    prediction = tf.nn.softmax(out)
    layer_names = [f"Hidden Layer {i+1} FC {dim}" for i, dim in enumerate(dims)]
    layer_names.append("Logits/Head")

# calculate the loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#, var_list=g_vars)

N = 100
test_images = mnist.test.images[:N]
test_labels = mnist.test.labels[:N]


for sim_measure in ["cka_kernel", "cka_linear", "rsa_euclidean", "rsa_cosine", "rsa_correlation"]:

    # init session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    start = sess.run([hidden1, hidden2, hidden3, hidden4, out],
                     feed_dict={x: test_images, y: test_labels})
    prev = start

    similarities = []
    similarities_prev = []
    steps = []
    all_representations = []
    labels = []
    colors = []


    for i in range(200):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})


        if i % 5  == 0:
            steps.append(i)
            representations = sess.run([hidden1, hidden2, hidden3, hidden4, out],
                                               feed_dict={x: test_images, y: test_labels})
            labels = labels + [f"{i} ({j+1})" for j in range(5)]
            colors = colors + list(range(5))
            peter = representations[0].reshape((N,-1))
            all_representations = all_representations + [r.reshape((N,-1)) for r in representations]

            similarities_of_step = [apply_representation_similarity(np.array([np.reshape(s, (N, -1)), np.reshape(r, (N, -1))]), sim_measure) for
                                    s, r in zip(start, representations)]
            similarities_of_step_prev = [apply_representation_similarity(np.array([np.reshape(s, (N, -1)), np.reshape(r, (N, -1))]), sim_measure)
                                         for s, r in zip(prev, representations)]

            # if sim_measure == "cka":
            #     similarities_of_step = [kernel_CKA(np.reshape(s, (N, -1)), np.reshape(r, (N, -1))) for s, r in zip(start, representations)]
            #     similarities_of_step_prev = [kernel_CKA(np.reshape(s, (N, -1)), np.reshape(r, (N, -1))) for s, r in zip(prev, representations)]
            #
            # else:
            #     print(np.mean(start[0]), np.mean(representations[0]))
            #     similarities_of_step = [rsa(np.array([np.reshape(s, (N, -1)), np.reshape(r, (N, -1))]), sim_measure) for
            #                             s, r in zip(start, representations)]
            #     similarities_of_step_prev = [rsa(np.array([np.reshape(s, (N, -1)), np.reshape(r, (N, -1))]), sim_measure)
            #                                  for s, r in zip(prev, representations)]

            similarities.append(similarities_of_step)
            similarities_prev.append(similarities_of_step_prev)

            prev = representations.copy()

            print(i, compute_accuracy(mnist.test.images, mnist.test.labels))

    #plot_rsa(all_representations, labels, colors)

    similarities = np.array(similarities).transpose()
    similarities_prev = np.array(similarities_prev).transpose()

    fig = plt.figure(figsize=(4, 2.5))
    methods = sim_measure.split("_")
    plt.xlabel("Number of training steps")

    #plt.yscale('symlog', linthreshy=0.015)
    plt.ylim(-0.05, 1.05)
    for i in range(len(similarities)):
        plt.plot(steps, similarities[i], label=layer_names[i])
        #plt.plot(range(len(similarities_prev[i])), similarities_prev[i], label=layer_names[i]+" to prev")

    if methods[0] == "cka":
        plt.title(f"CKA ({methods[1]}) similarity")
        plt.ylabel("Similarity")
        plt.legend(loc="lower left")
    elif methods[0] == "rsa":
        plt.title(f"RSA ({methods[1]}) dissimilarity")
        plt.ylabel("Dissimilarity")
        plt.legend(loc="upper left")


    #plt.show()
    plt.savefig(f"img/mnist_{sim_measure}.pdf")


