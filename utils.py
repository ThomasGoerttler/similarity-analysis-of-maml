""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

import matplotlib.pyplot as plt
from cka import linear_CKA, kernel_CKA, cka_rdm
from rsa import rsa, mds

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

        if int(iteration) >= 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(iteration)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

def reshape_elems_of_list(layers, shape = (-1, -1)):
    reshaped_layers = []
    for layer in layers:
        layer = np.reshape(layer, shape)
        reshaped_layers.append(layer)
    return reshaped_layers
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

def plot_neighbour_analysis(steps, similaritiy_to_pre, layer_names, method):

    print(similaritiy_to_pre)
    fig = plt.figure(figsize=(6, 4))
    plt.xlabel("Number of training steps")


    #plt.yscale('symlog', linthreshy=0.015)
    print(np.array(steps)[1:])
    for i in range(5):
        plt.plot(np.array(steps)[1:], similaritiy_to_pre[i], label=layer_names[i])
        print(similaritiy_to_pre[i])
    if method.startswith("rsa"):
        plt.suptitle("Dissimilarity between meta-initilized representation")
        plt.title("and meta-initilized representation from 1000 training steps before")
        plt.ylabel("Dissimilarity")
        plt.legend(loc="upper right")
    else:
        plt.suptitle("Similarity between meta-initilized representation")
        plt.title("and meta-initilized representation from 1000 training steps before")
        plt.ylabel("Similarity")
        plt.legend(loc="lower right")

    plt.savefig(f"img/neighbour_analysis_{method}.pdf")
    plt.show()


def plot_base_analysis(steps, final_mean_diff_to_base, final_sem_diff_to_base, layer_names, method):
    fig = plt.figure(figsize=(6, 4))
    plt.xlabel("Number of training steps of meta-learning")
    for i in range(5):
        x = np.array(steps)
        y = np.array(final_mean_diff_to_base[i])
        sem = np.array(final_sem_diff_to_base[i])

        # Calculate upper and lower bounds of the confidence interval
        upper_bound = y + 1.96 * sem  # You can adjust the multiplier (e.g., 1.96 for 95% confidence interval)
        lower_bound = y - 1.96 * sem  # You can adjust the multiplier (e.g., 1.96 for 95% confidence interval)

        # Plot shaded region between upper and lower bounds
        plt.fill_between(x, lower_bound, upper_bound, alpha=0.3)

        plt.plot(x, y, label=layer_names[i])

    if method.startswith("rsa"):
        plt.suptitle("Mean dissimilarity between fine-tuned representation")
        plt.title("and meta-initialized representation")
        plt.ylabel("Dissimilarity")
        plt.legend(loc="upper left")
        plt.ylim(-0.05, 1.05)

    else:
        plt.suptitle("Mean similarity between fine-tuned representation")
        plt.title("and meta-initialized representation")
        plt.ylabel("Similarity")
        plt.legend(loc="lower left")
        plt.ylim(-0.05, 1.05)

    plt.savefig(f"img/base_analysis_{method}.pdf")
    plt.show()

def plot_performance(steps, accs, task_to_analyze):

    acc_singles = [[] for _ in range(task_to_analyze)]
    acc_sum = []
    acc_sem = []
    for acc in accs:
        for i in range(task_to_analyze):
            acc_singles[i].append(acc[i + 1])  # First is cka evaluation task
        acc_sum.append(np.mean(acc))
        acc_sem.append(np.std(acc) / np.sqrt(len(acc)))
    fig = plt.figure(figsize=(4, 4))
    plt.title("Performance of four selected tasks")
    for i in range(task_to_analyze):
        plt.plot(steps, acc_singles[i], color=DISTINCT_COLORS[i], label=f"Task {i+1}")

    acc_sum = np.array(acc_sum)
    acc_sem = np.array(acc_sem)
    # Calculate upper and lower bounds of the confidence interval
    upper_bound = acc_sum + 1.96 * acc_sem
    lower_bound = acc_sum - 1.96 * acc_sem

    # Plot shaded region between upper and lower bounds
    plt.fill_between(steps, lower_bound, upper_bound, color="black", alpha=0.3)

    plt.plot(steps, acc_sum, color="black", label=f"Average over {len(accs[0])} tasks")
    plt.xlabel("Number of training steps")
    plt.ylabel("Accuracy")# after 5 inner gradient steps")
    plt.legend(loc="lower right")

    plt.savefig("img/performance_of_four_selected_tasks.pdf")
    plt.show()


def apply_representation_similarity(array, method):

    if method == "rsa_correlation":
        return rsa(array, "correlation")
    elif method == "rsa_euclidean":
        return rsa(array, "euclidean")
    elif method == "rsa_cosine":
        return rsa(array, "cosine")
    elif method == "cka_linear":
        if len(array) == 2:
            return linear_CKA(array[0], array[1])
        else:
            rdms = cka_rdm(array, linear_CKA)
            return rdms
    elif method == "cka_kernel":
        if len(array) == 2:
            return kernel_CKA(array[0], array[1])
        else:
            rdms = cka_rdm(array, kernel_CKA)
            return rdms


from mpl_toolkits.axes_grid1 import make_axes_locatable

class AxesDecorator():
    def __init__(self, ax, size="5%", pad=0.05, ticks=[1,2,3], ticks_label = [1,2,3], spacing=0.05,
                 color="k"):
        self.divider= make_axes_locatable(ax)
        self.ax = self.divider.new_vertical(size=size, pad=pad, sharex=ax, pack_start=True)
        ax.figure.add_axes(self.ax)
        self.ticks=np.array(ticks)
        self.d = np.mean(np.diff(ticks))
        self.spacing = spacing
        self.get_curve()
        self.color=color
        for x0 in ticks:
            self.plot_curve(x0)
        self.ax.set_yticks([])
        plt.setp(ax.get_xticklabels(), visible=False)
        self.ax.tick_params(axis='x', which=u'both',length=0)
        ax.tick_params(axis='x', which=u'both',length=0)
        for direction in ["left", "right", "bottom", "top"]:
            self.ax.spines[direction].set_visible(False)
        self.ax.set_xlabel(ax.get_xlabel())
        ax.set_xlabel("")
        self.ax.set_xticks(self.ticks)
        self.ax.set_xticklabels(ticks_label)
        print(ticks_label)

    def plot_curve(self, x0):
        x = np.linspace(x0-self.d/2.*(1-self.spacing),x0+self.d/2.*(1-self.spacing), 50 )
        self.ax.plot(x, self.curve, c=self.color)

    def get_curve(self):
        lx = np.linspace(-np.pi/2.+0.05, np.pi/2.-0.05, 25)
        tan = np.tan(lx)*10
        self.curve = np.hstack((tan[::-1],tan))
        return self.curve


def extract_representation(layers, layer_names, meta_steps, num_analysis_tasks, method, inner_steps = [0, 1, 5, 10]):


    final_base_representation = []
    final_mean_diff_to_base = []
    final_sem_diff_to_base = []
    final = []
    final_colors = []

    for layer_id in range(len(layer_names)):

        representations = []
        base_representations = []
        mean_diff_to_base = []
        sem_diff_to_base = []
        colors = []


        for meta_step_id, meta_step in enumerate(meta_steps):
            # the first zero acces the first task, while the second zero means zeros innersteps
            task_0_representations = layers[layer_id]

            tt = task_0_representations[meta_step_id][0]
            base_representations.append(tt[0])  # before any inner loop
            diff_to_base = []
            for task_id in range(num_analysis_tasks):
                # get representations of specific task for all inner steps
                representations = representations + [layers[layer_id][meta_step_id][task_id][step] for step in
                                                     inner_steps]

                diff_to_base.append(
                    apply_representation_similarity(np.array([base_representations[-1], representations[-1]]),
                                                    method=method))
                colors = colors + [task_id + meta_step_id * num_analysis_tasks] * len(inner_steps)
            mean_diff_to_base.append(np.mean(diff_to_base))
            sem_diff_to_base.append(np.std(diff_to_base) / np.sqrt(len(diff_to_base)))

        final_colors.append(colors)
        final.append(np.array(representations))
        final_base_representation.append(np.array(base_representations))
        final_mean_diff_to_base.append(mean_diff_to_base)
        final_sem_diff_to_base.append(sem_diff_to_base)

    return final, final_base_representation, final_mean_diff_to_base, final_sem_diff_to_base, final_colors


import numpy as np


def extract_mean_across_blocks(matrix, block_size, sub_block_size, num_blocks):
    """
    Extracts the first row from each diagonal sub_block_size x sub_block_size block within block_size x block_size blocks,
    and calculates the mean of corresponding elements across all first rows.

    Parameters:
    matrix (numpy.ndarray): A matrix of shape (num_blocks * block_size, num_blocks * block_size)
    block_size (int): The size of the main blocks (e.g., 32 for a 32x32 block)
    sub_block_size (int): The size of the diagonal sub-blocks (e.g., 16 for a 16x16 block)
    num_blocks (int): The number of main blocks

    Returns:
    list: A list containing the mean of corresponding elements from the first rows of diagonal sub-blocks across all blocks.
    """
    matrix_size = matrix.shape[0]

    result = []

    # Loop through each of the main blocks
    for block_idx in range(num_blocks):
        # Determine where this block starts in the matrix
        block_start_row = block_idx * block_size
        block_start_col = block_idx * block_size

        # Ensure we do not exceed matrix bounds
        if block_start_row + block_size > matrix_size or block_start_col + block_size > matrix_size:
            raise ValueError(f"Block size exceeds matrix dimensions at block {block_idx}.")

        # Store first rows from diagonal sub-blocks within the current block
        first_rows = []

        # Loop through the diagonal sub-blocks in the current block
        for sub_block_idx in range(0, block_size, sub_block_size):
            row_start = block_start_row + sub_block_idx
            col_start = block_start_col + sub_block_idx

            # Ensure we are not going out of bounds in sub-blocks
            if row_start >= matrix_size or col_start + sub_block_size > matrix_size:
                raise ValueError(f"Sub-block index exceeds matrix dimensions at row {row_start}, col {col_start}.")

            # Extract the first row of the current sub-block
            first_row = matrix[row_start, col_start:col_start + sub_block_size]
            first_rows.append(first_row)

        # Convert to a NumPy array for easier element-wise operations
        first_rows = np.array(first_rows)

        # Calculate the mean of the first, second, ..., elements across the first rows in this block
        block_mean = first_rows.mean(axis=0)
        result.append(block_mean)

    # Convert the result to a list of lists
    return [mean.tolist() for mean in result]


from itertools import product


def extract_linked_indices(matrix, block_size):
    """
    Extracts linked elements from the given matrix based on a specified block size.

    Args:
    matrix (numpy.ndarray): A matrix (e.g., 80x80).
    block_size (int): The size of the block to determine the linked indices.

    Returns:
    list: A list of all values for the linked index combinations.
    """
    # Generate starting rows/cols based on block size
    linked_indices = [i for i in range(0, matrix.shape[0], block_size + 1)]

    # Get all combinations of (row, col) indices
    index_combinations = list(product(linked_indices, repeat=2))

    # Filter out-of-bounds combinations (just in case, but should be within bounds)
    valid_combinations = [(row, col) for row, col in index_combinations if
                          row < matrix.shape[0] and col < matrix.shape[1]]

    # Extract values from the matrix for all valid combinations
    linked_elements = [matrix[row, col] for row, col in valid_combinations]

    return linked_elements
def plot_rsa_fancy(activations_per_model, labels = None, method = "cka_linear", n_tasks = 1, title = "", steps=None):

    n_steps = len(steps)
    fig, axes = plt.subplots(1, 2, figsize = (8,4))
    r = apply_representation_similarity(activations_per_model, method)
    #r_old = rsa(activations_per_model, "correlation")

    #print(extract_mean_across_blocks(r, block_size=16, sub_block_size=4, num_blocks=2))
    #print(n_tasks*len(steps), n_tasks, len(labels))
    print(title, extract_mean_across_blocks(r, block_size=int(len(labels)/len(steps)), sub_block_size=int(len(labels)/len(steps)/n_tasks), num_blocks=int(len(steps))))

    print(extract_linked_indices(r, block_size=int(len(labels)/len(steps))))

    mat = axes[0].imshow(r)
    if title != "":
        plt.suptitle(title)

    axes[0].title.set_text("Representational similarity matrix")
    axes[0].set_xticks(np.array(range(len(steps))) * 12 + 5.5)
    axes[0].set_yticks([])
    axes[0].set_xticklabels(steps, rotation='vertical')
    n_per_step = len(activations_per_model) / n_steps

    AxesDecorator(axes[0], ticks=np.array(range(len(steps))) * n_per_step + (n_per_step-1)/2.0, ticks_label = steps)
    #AxesDecorator(axes[0], ticks=np.array(range(3)) * n_per_step + (n_per_step-1)/2.0, ticks_label = steps)
    #axes[0].set_yticklabels()
    import matplotlib.ticker as tick
    fig.colorbar(mat, ax=axes[0], fraction=0.046, pad=0.04, format=tick.FormatStrFormatter('%.3f'))

    m = mds(r)
    axes[1].title.set_text("Multidimensional scaling")
    n_per_task_per_step = len(activations_per_model) / n_tasks / n_steps
    for j in range(n_tasks * len(steps)):
        base = int(j*n_per_task_per_step)
        fine_tunes = slice(int(j*n_per_task_per_step+1), int((j+1)*n_per_task_per_step))
        fine_tune_lines = slice(int(j*n_per_task_per_step), int((j+1)*n_per_task_per_step))

        axes[1].scatter(m[base][0], m[base][1], label = labels, c = "#000000", s = 40, marker = "*")

        if steps != None:
            axes[1].annotate(steps[int(j/n_tasks)], (m[base][0], m[base][1]), xytext=(m[base][0]*0.85+0.2, m[base][1]*0.85+0.2))

        axes[1].plot(m[fine_tune_lines][:,0], m[fine_tune_lines][:,1], c = "#000000", ls = '--', linewidth=0.25)

        axes[1].scatter(m[fine_tunes][:,0], m[fine_tunes][:,1], label = labels, c = DISTINCT_COLORS[j%n_tasks], s = 20, edgecolor='black', linewidth=0.25)

        max = np.max(np.absolute(m))
        max = max * 1.1

        axes[1].set_xlim(-max, max)
        axes[1].set_ylim(-max, max)
        axes[1].set_aspect('equal')
    #axes[1].scatter(m[0,0], m[0,1], label=labels, c="#000000", s=30)

    axes[1].set_xlabel("MDS 1")
    axes[1].set_ylabel("MDS 2")
    #for i, txt in enumerate(labels):
        #axes[1].annotate(txt, (m[i,0], m[i,1]))
    plt.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96, wspace=0.5)
    plt.savefig(f"img/fancy_{title}.pdf")
    plt.show()
