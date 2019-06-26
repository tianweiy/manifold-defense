"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import math

from model import Model
import cifar10_input
from pgd_attack import L2PGDAttack, LinfPGDAttack
from attacks import L2OPAttack, ce


with open('config.json') as config_file:
    config = json.load(config_file)

# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

# Setting up training parameters
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
weight_decay = config['weight_decay']
data_path = config['data_path']
momentum = config['momentum']
batch_size = config['training_batch_size']
op_weight = config['overpowered_weight']

# Setting up the data and the model
raw_cifar = cifar10_input.CIFAR10Data(data_path)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(mode='train')
model_file = tf.train.latest_checkpoint(config['model_dir'])

# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)
total_loss = model.mean_xent + weight_decay * model.weight_decay_loss
opt = tf.train.MomentumOptimizer(learning_rate, momentum)
train_step = opt.minimize(
    total_loss,
    global_step=global_step)

# Set up adversary
l2_attack = L2PGDAttack(model,
                     config['epsilon'],
                     config['num_steps'],
                     config['step_size'],
                     config['random_start'],
                     config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
# saver = tf.train.import_meta_graph("./models/adv_trained/checkpoint-1000.meta.tmpf3ac6c9416514c0698d28ea156ebf732")
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv', model.xent / batch_size)
tf.summary.image('images adv train', model.x_input)
merged_summaries = tf.summary.merge_all()

# keep the configuration file with the model for reproducibility
shutil.copy('config.json', model_dir)

with tf.Session() as sess:
    # load adv-trained checkpoint
    # Restore the checkpoint
    saver.restore(sess, model_file)
    gan = hub.Module("https://tfhub.dev/google/compare_gan/model_13_cifar10_resnet_cifar/1")

    # overpowered attack
    op_attack = L2OPAttack(gan, model)

    # initialize data augmentation
    cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)

    # Initialize the summary writer, global variables, and our time counter.
    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)

    # saver.restore(sess, "./models/adv_trained/checkpoint")
    sess.run(tf.global_variables_initializer())
    training_time = 0.0

    # Main training loop
    for ii in range(max_num_training_steps):
        x_batch, y_batch = cifar.train_data.get_next_batch(batch_size,
                                                           multiple_passes=True)
        # normal situation:
        if ii % 391 == 0:
            # overpowered attack:
            batch1, batch2, is_adv = op_attack.perturb(sess)
            op_loss = op_weight * tf.reduce_sum(ce(model._forward(batch1), model._forward(batch2)) * is_adv) / tf.reduce_sum(is_adv)
            op_step = opt.minimize(op_loss)
            sess.run(op_step)

        # Compute Adversarial Perturbations
        start = timer()
        x_batch_adv = l2_attack.perturb(x_batch, y_batch, sess)
        end = timer()

        training_time += end - start

        nat_dict = {model.x_input: x_batch,
                    model.y_input: y_batch}

        adv_dict = {model.x_input: x_batch_adv,
                    model.y_input: y_batch}

        # Output to stdout
        if ii % num_output_steps == 0:
            nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
            adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
            print('Step {}:    ({})'.format(ii, datetime.now()))
            print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
            print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
            if ii != 0:
                print('    {} examples per second'.format(
                    num_output_steps * batch_size / training_time))
                training_time = 0.0
        # Tensorboard summaries
        if ii % num_summary_steps == 0:
            summary = sess.run(merged_summaries, feed_dict=adv_dict)
            summary_writer.add_summary(summary, global_step.eval(sess))

        # Write a checkpoint
        if ii % num_checkpoint_steps == 0:
            saver.save(sess,
                       os.path.join(model_dir, 'checkpoint'),
                       global_step=global_step)

        # Actual training step
        start = timer()
        sess.run(train_step, feed_dict=adv_dict)
        end = timer()
        training_time += end - start

        # evaluate every epoch
        if ii % 400 == 0:
            # evalaution on test set
            num_eval_examples = 10000
            eval_batch_size = 100
            total_corr = 0

            num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

            # first construct adv example
            x_adv = []  # adv accumulator

            print('Iterating over {} batches'.format(num_batches))

            for ibatch in range(num_batches):
                bstart = ibatch * eval_batch_size
                bend = min(bstart + eval_batch_size, num_eval_examples)
                print('batch size: {}'.format(bend - bstart))

                x_batch = cifar.eval_data.xs[bstart:bend, :]
                y_batch = cifar.eval_data.ys[bstart:bend]

                x_batch_adv = l2_attack.perturb(x_batch, y_batch, sess)

                x_adv.append(x_batch_adv)

            # then evaluate the accuracy
            # Iterate over the samples batch-by-batch
            for ibatch in range(num_batches):
                bstart = ibatch * eval_batch_size
                bend = min(bstart + eval_batch_size, num_eval_examples)

                x_batch = x_adv[bstart:bend, :]
                y_batch = cifar.eval_data.ys[bstart:bend]

                dict_adv = {model.x_input: x_batch,
                            model.y_input: y_batch}
                cur_corr, y_pred_batch = sess.run([model.num_correct, model.predictions],
                                                  feed_dict=dict_adv)

                total_corr += cur_corr

            accuracy = total_corr / num_eval_examples
            print('    test adv accuracy {:.4}%'.format(accuracy * 100))