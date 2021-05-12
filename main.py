"""
Usage Instructions:
    10-shot sinusoid:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10
    10-shot sinusoid baselines:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10 --baseline=oracle
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10
    5-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=32 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot5way/
    20-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=logs/omniglot20way/
    5-way 1-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet1shot/ --num_filters=32 --max_pool=True
    5-way 5-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=5 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet5shot/ --num_filters=32 --max_pool=True
    To run evaluation, use the '--train=False' flag and the '--test_set=True' flag to use the test set.
    For omniglot and miniimagenet training, acquire the dataset online, put it in the correspoding data directory, and see the python script instructions in that directory to preprocess the data.
    Note that better sinusoid results can be achieved by using a larger network.
"""
import csv
import numpy as np
import pickle
import random
import tensorflow as tf

from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags

from utils import load_model, interpret_steps, plot_neighbour_analysis, plot_base_analysis, plot_performance

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

## Analysis parameters
flags.DEFINE_bool('analyze', False, 'True to analyze, False to not analyze.')
flags.DEFINE_integer('points_to_analyze', 4, 'number of points to analyze Default: 4')
flags.DEFINE_bool('base_analysis', False, 'In case you specifically want to analyze the meta-optimization')
flags.DEFINE_string('steps_to_analyze', "range(0,1,1)", 'Steps to analyze. Can be a list, a range, or a number')

def train(model, saver, sess, exp_string, data_generator, resume_itr=1):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    if FLAGS.datasource == 'sinusoid':
        PRINT_INTERVAL = 1000
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    else:
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []

    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations + 1): # as resume_itr starts at 1, we have to +1
        feed_dict = {}
        if 'generate' in dir(data_generator):
            batch_x, batch_y, amp, phase = data_generator.generate()

            if FLAGS.baseline == 'oracle':
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                for i in range(FLAGS.meta_batch_size):
                    batch_x[i, :, 1] = amp[i]
                    batch_x[i, :, 2] = phase[i]

            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :] # b used for testing
            labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}

        if itr < FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            if model.classification:
                input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])

        result = sess.run(input_tensors, feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-2])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr), write_meta_graph=False)

        # sinusoid is infinite data, so no need to test on meta-validation set.
        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0 and FLAGS.datasource !='sinusoid':
            if 'generate' not in dir(data_generator):
                feed_dict = {}
                if model.classification:
                    input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates-1], model.summ_op]
                else:
                    input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1], model.summ_op]
            else:
                batch_x, batch_y, amp, phase = data_generator.generate(train=False)
                inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
                inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :]
                labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
                labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
                feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}
                if model.classification:
                    input_tensors = [model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]]
                else:
                    input_tensors = [model.total_loss1, model.total_losses2[FLAGS.num_updates-1]]

            result = sess.run(input_tensors, feed_dict)
            print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))

# calculated for omniglot
NUM_TEST_POINTS = 600

def test(model, saver, sess, exp_string, data_generator, test_num_updates=None):
    num_classes = data_generator.num_classes # for classification, 1 otherwise

    np.random.seed(1)
    random.seed(1)


    steps = range(1000,61000,1000)
    accs = []

    inner_loops = 5
    for step in steps:
        print(f"Load model {step}")
        load_model(FLAGS.logdir, exp_string, saver, sess, step)

        metaval_accuracies = []

        for _ in range(NUM_TEST_POINTS):
            if 'generate' not in dir(data_generator):
                feed_dict = {}
                feed_dict = {model.meta_lr : 0.0}
            else:
                batch_x, batch_y, amp, phase = data_generator.generate(train=False)

                if FLAGS.baseline == 'oracle': # NOTE - this flag is specific to sinusoid
                    batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                    batch_x[0, :, 1] = amp[0]
                    batch_x[0, :, 2] = phase[0]

                inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
                inputb = batch_x[:,num_classes*FLAGS.update_batch_size:, :]
                labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
                labelb = batch_y[:,num_classes*FLAGS.update_batch_size:, :]

                feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}

            if model.classification:
                result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
            else:  # this is for sinusoid
                result = sess.run([model.total_loss1] +  model.total_losses2, feed_dict)
            metaval_accuracies.append(result[inner_loops])

        accs.append(np.array(metaval_accuracies))

    plot_performance(steps, accs, NUM_TEST_POINTS)


    # means = np.mean(metaval_accuracies, 0)
    # stds = np.std(metaval_accuracies, 0)
    # ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)
    #
    # print('Mean validation accuracy/loss, stddev, and confidence intervals')
    # print((means, stds, ci95))
    #
    # out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv'
    # out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.pkl'
    # with open(out_pkl, 'wb') as f:
    #     pickle.dump({'mses': metaval_accuracies}, f)
    # with open(out_filename, 'w') as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerow(['update'+str(i) for i in range(len(means))])
    #     writer.writerow(means)
    #     writer.writerow(stds)
    #     writer.writerow(ci95)


def analyze(model, saver, sess, exp_string, data_generator, test_num_updates=None, NUM_ANALYSIS_POINTS=1, base_analysis=False, steps = [-1]):

    ### computing activations

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    np.random.seed(1)
    random.seed(1)

    print(exp_string)
    hid1, hid2, hid3, hid4, out, acc = [], [], [], [], [], []

    for step in steps:
        meta_hidden1s = []
        meta_hidden2s = []
        meta_hidden3s = []
        meta_hidden4s = []
        meta_outputs = []
        metaval_accuracies = []
        print(f"Load model {step}")
        load_model(FLAGS.logdir, exp_string, saver, sess, step)
        print(f"Load model {step} done!")
        for i in range(NUM_ANALYSIS_POINTS+1):

            if i==0: # The first sample is the evaluation sample
                continue;

            if 'generate' not in dir(data_generator):
                feed_dict = {}
                feed_dict = {model.meta_lr : 0.00}
            else:
                batch_x, batch_y, amp, phase = data_generator.generate(train=False)

                if FLAGS.baseline == 'oracle': # NOTE - this flag is specific to sinusoid
                    batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                    batch_x[0, :, 1] = amp[0]
                    batch_x[0, :, 2] = phase[0]

                inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
                inputb = batch_x[:,num_classes*FLAGS.update_batch_size:, :]
                labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
                labelb = batch_y[:,num_classes*FLAGS.update_batch_size:, :]

                feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}

            targets = [model.hiddens1, model.hiddens2, model.hiddens3, model.hiddens4, model.outputs, model.metaval_total_accuracy1 + model.metaval_total_accuracies2]

            def reshape_elems_of_list(layers, shape = (model.dim_output, -1)):
                reshaped_layers = []
                for layer in layers:
                    layer = np.reshape(layer, shape)
                    reshaped_layers.append(layer)
                return reshaped_layers

            hidden1s, hidden2s, hidden3s, hidden4s, outputs, a = sess.run(targets, feed_dict)

            meta_hidden1s.append(reshape_elems_of_list(hidden1s))
            meta_hidden2s.append(reshape_elems_of_list(hidden2s))
            meta_hidden3s.append(reshape_elems_of_list(hidden3s))
            meta_hidden4s.append(reshape_elems_of_list(hidden4s))
            meta_outputs.append(reshape_elems_of_list(outputs))
            metaval_accuracies.append(a)

        hid1.append(meta_hidden1s)
        hid2.append(meta_hidden2s)
        hid3.append(meta_hidden3s)
        hid4.append(meta_hidden4s)
        out.append(meta_outputs)
        acc.append(metaval_accuracies)

    ### prepare for visualizing
    from rsa import plot_rsa_fancy, rsa
    layers = [hid1, hid2, hid3, hid4, out]
    if FLAGS.datasource == 'miniimagenet':
        layer_names = ["Pooling layer 1", "Pooling layer 2", "Pooling layer 3", "Pooling layer 4", "Logits/Head"]
    else:
        layer_names = ["Convolution layer 1", "Convolution layer 2", "Convolution layer 3", "Convolution layer 4", "Logits/Head"]

    final_base_representation = []
    final_mean_diff_to_base = []
    final_std_diff_to_base = []
    for i, (layer_name) in enumerate(layer_names):

        representations = []
        base_representations = []
        mean_diff_to_base = []
        std_diff_to_base = []
        labels = []
        colors = []
        inner_steps = [0, 1, 5, 10]

        for j, step in enumerate(steps):
            base_representations.append(layers[i][j][0][0])
            diff_to_base = []
            for k in range(NUM_ANALYSIS_POINTS):
                representations = representations + list(map(layers[i][j][k].__getitem__, inner_steps))
                diff_to_base.append(rsa(np.array([base_representations[-1], representations[-1]])))
                colors = colors + [k + j*NUM_ANALYSIS_POINTS] * len(inner_steps)
            mean_diff_to_base.append(np.mean(diff_to_base))
            std_diff_to_base.append(np.std(diff_to_base))
            labels = colors

        final = np.array(representations)
        final_base_representation.append(np.array(base_representations))
        final_mean_diff_to_base.append(np.array(mean_diff_to_base))
        final_std_diff_to_base.append(np.array(std_diff_to_base))

        if not base_analysis:
            plot_rsa_fancy(final, labels, colors, method="correlation", title=layer_name, n_tasks=NUM_ANALYSIS_POINTS, steps=steps)

    if base_analysis:
        plot_neighbour_analysis(steps, final_base_representation, layer_names)
        plot_base_analysis(steps, final_mean_diff_to_base, final_std_diff_to_base, layer_names)


def main():
    if FLAGS.datasource == 'sinusoid':
        if FLAGS.train:
            test_num_updates = 5
        else:
            test_num_updates = 10
    else:
        if FLAGS.datasource == 'miniimagenet':
            if FLAGS.train == True:
                test_num_updates = 1  # eval on at least one update during training
            else:
                test_num_updates = 10
        else:
            test_num_updates = 10

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1

    if FLAGS.datasource == 'sinusoid':
        data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)
    else:
        if FLAGS.metatrain_iterations == 0 and FLAGS.datasource == 'miniimagenet':
            assert FLAGS.meta_batch_size == 1
            assert FLAGS.update_batch_size == 1
            data_generator = DataGenerator(1, FLAGS.meta_batch_size)  # only use one datapoint,
        else:
            if FLAGS.datasource == 'miniimagenet': # TODO - use 15 val examples for imagenet?
                if FLAGS.train:
                    data_generator = DataGenerator(FLAGS.update_batch_size+15, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
                else:
                    data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
            else:
                data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory


    dim_output = data_generator.dim_output
    if FLAGS.baseline == 'oracle':
        assert FLAGS.datasource == 'sinusoid'
        dim_input = 3
        FLAGS.pretrain_iterations += FLAGS.metatrain_iterations
        FLAGS.metatrain_iterations = 0
    else:
        dim_input = data_generator.dim_input

    if FLAGS.datasource == 'miniimagenet' or FLAGS.datasource == 'omniglot':
        tf_data_load = True
        num_classes = data_generator.num_classes

        if FLAGS.train: # only construct training model if needed
            random.seed(5)
            image_tensor, label_tensor = data_generator.make_data_tensor()
            inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
            inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
            labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
            labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
            input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

        random.seed(6)
        image_tensor, label_tensor = data_generator.make_data_tensor(train=False, shuffle = False, analysis=FLAGS.analyze, points_to_analyze = FLAGS.points_to_analyze)
        inputa = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
        inputb = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
        labela = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
        labelb = tf.slice(label_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
        metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

    else:
        tf_data_load = False
        input_tensors = None

    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    if FLAGS.train or not tf_data_load:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    if tf_data_load:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep= 100)

    sess = tf.InteractiveSession()

    if FLAGS.train == False:
        # change to original meta batch size after loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)

    if FLAGS.num_filters != 64:
        exp_string += 'hidden' + str(FLAGS.num_filters)
    if FLAGS.max_pool:
        exp_string += 'maxpool'
    if FLAGS.stop_grad:
        exp_string += 'stopgrad'
    if FLAGS.baseline:
        exp_string += FLAGS.baseline
    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif FLAGS.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')

    resume_itr = 1 # We start with 1
    model_file = None

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    load_model(FLAGS.logdir, exp_string, saver, sess, FLAGS.test_iter)

    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        if FLAGS.analyze:
            analyze(model, saver, sess, exp_string, data_generator, test_num_updates, FLAGS.points_to_analyze, base_analysis=FLAGS.base_analysis, steps=interpret_steps(FLAGS.steps_to_analyze))
        else:
            test(model, saver, sess, exp_string, data_generator, test_num_updates)

if __name__ == "__main__":
    main()
