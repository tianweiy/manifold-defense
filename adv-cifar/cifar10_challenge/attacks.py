"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

def ce(y_, y):
    return tf.reduce_sum(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[-1]))


class L2OPAttack:
    def __init__(self):
        """overpowered attack using max-min optimization"""

    def perturb(self, sess, net, gan, eps, embed_feats=128, num_images=128, z_lr=5e-3, lambda_lr=1e-4,
             num_steps=1000):

        BATCH_SIZE = 64

        batch1 = tf.zeros((num_images, 3, 32, 32))
        batch2 = tf.zeros((num_images, 3, 32, 32))
        is_valid = tf.zeros(num_images)
        EPS = eps

        for i in range(num_images // BATCH_SIZE):
            # sample two latent code
            z1 = tf.Variable(tf.random.uniform([BATCH_SIZE, embed_feats]))
            z2 = tf.Variable(tf.random.uniform([BATCH_SIZE, embed_feats]))

            ones = tf.ones(z1.shape[0])

            lambda_ = tf.Variable(1e0 * tf.ones([z1.shape[0], 1]))

            opt1 = tf.train.MomentumOptimizer(z_lr, momentum=0.9)
            opt2 = tf.train.MomentumOptimizer(lambda_lr, momentum=0.9)


            for j in range(num_steps):
                print("Steps: ", j)
                # generate images
                x1 = gan(z1, signature="generator")
                x2 = gan(z2, signature="generator")

                distance_mat = tf.norm(tf.reshape(x1 - x2, (x1.shape[0], -1)), axis=-1, keep_dims=False) - EPS * ones

                # pre-softmax logits
                net_pre1 = tf.nn.softmax(net.forward(x1))
                net_pre2 = tf.nn.softmax(net.forward(x2))

                # predictecd labels
                net_res1 = tf.argmax(net_pre1, axis=-1)
                net_res2 = tf.argmax(net_pre2, axis=-1)

                is_adv = tf.cast(1 - (net_res1 == net_res2), tf.float32)
                is_feasible = tf.cast((distance_mat <= 0), tf.float32)
                not_valid = tf.cast(1 - (is_adv * is_feasible), tf.float32)
                if tf.reduce_sum(is_adv * is_feasible) == BATCH_SIZE:

                    batch1[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, ...] = x1
                    batch2[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, ...] = x2
                    is_valid[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = 1.
                    break

                # calculate loss1, update opt1
                loss1 = (-1.* tf.reduce_sum(ce(net_pre1, net_pre2), reduction_indices=None)*not_valid) +\
                        tf.reduce_sum(lambda_ * distance_mat*not_valid) + 1e-4*tf.reduce_sum(tf.norm(z1, axis=-1)*not_valid) +\
                        1e-4*tf.reduce_sum(tf.norm(z2,axis=-1)*not_valid)/tf.reduce_sum(not_valid)

                opt_step1 = opt1.minimize(loss1, var_list=[z1, z2])

                loss2 = -1. * tf.reduce_mean(lambda_ * distance_mat * not_valid)
                opt_step2 = opt2.minimize(loss2, var_list=[lambda_])

                if i == 0 and j == 0:
                    sess.run(tf.global_variables_initializer())

                # update latent code
                sess.run([opt_step1, opt_step2])
                print(sess.run([net_res1, net_res2]))
                print(sess.run([-tf.reduce_sum(net_pre1 * tf.log(net_pre2), reduction_indices=[-1])]))
                print(sess.run(tf.reduce_sum(distance_mat)))

            batch1[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, ...] = x1
            batch2[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, ...] = x2
            is_valid[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = is_adv * is_feasible

        count = tf.reduce_sum(is_valid)
        print('number of adversarial pairs found:%d\n' % (count))

        return batch1, batch2, is_valid


if __name__ == '__main__':
    import json
    import sys
    import os
    import math

    from model import Model

    with open('config.json') as config_file:
        config = json.load(config_file)

    model_file = tf.train.latest_checkpoint(config['model_dir'])
    if model_file is None:
        print('No model found')
        sys.exit()

    model = Model(mode='eval')
    attack = L2OPAttack()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore the checkpoint
        saver.restore(sess, model_file)

        gan = hub.Module("https://tfhub.dev/google/compare_gan/model_13_cifar10_resnet_cifar/1")

        batch1, batch2, is_valid = attack.perturb(sess, model, gan, config['epsilon'])

        print('Storing examples')
        path = config['store_adv_path']
        batch1 = np.concatenate(batch1, axis=0)
        np.save(os.path.join(path, "batch1.npy"), batch1)

        batch2 = np.concatenate(batch2, axis=0)
        np.save(os.path.join(path, "batch2.npy"), batch2)
        print('Examples stored in {}'.format(path))
