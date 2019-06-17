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

embed_feats=128
BATCH_SIZE = 6
z_lr=5e-3
lambda_lr=1e-4
EPS = 255.0

def ce(y_, y):
    return tf.reduce_sum(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[-1]))


class L2OPAttack:
    def __init__(self, gan, net):
        """overpowered attack using max-min optimization"""
        self.z1 = tf.Variable(tf.random.uniform([BATCH_SIZE, embed_feats]), trainable=False)
        self.z2 = tf.Variable(tf.random.uniform([BATCH_SIZE, embed_feats]), trainable=False)
        self.lambda_ = tf.Variable(1e0 * tf.ones([self.z1.shape[0], 1]), trainable=False)

        self.opt1 = tf.train.MomentumOptimizer(z_lr, momentum=0.9)
        self.opt2 = tf.train.MomentumOptimizer(lambda_lr, momentum=0.9)

        self._build_graph(gan, net)

    def _build_graph(self, gan, net):
        """Build the computation graph for generating attacks"""
        ones = tf.ones(self.z1.shape[0])
        # generate images
        self.x1 = gan(self.z1, signature="generator")
        self.x2 = gan(self.z2, signature="generator")

        self.distance_mat = tf.norm(tf.reshape(self.x1 - self.x2, (self.x1.shape[0], -1)), axis=-1, keep_dims=False) - EPS * ones

        # pre-softmax logits
        self.net_pre1 = tf.nn.softmax(net.forward(self.x1))
        self.net_pre2 = tf.nn.softmax(net.forward(self.x2))

        # predictecd labels
        net_res1 = tf.argmax(self.net_pre1, axis=-1)
        net_res2 = tf.argmax(self.net_pre2, axis=-1)

        self.is_adv = tf.cast(1 - (net_res1 == net_res2), tf.float32)
        self.is_feasible = tf.cast((self.distance_mat <= 0), tf.float32)
        self.not_valid = tf.cast(1 - (self.is_adv * self.is_feasible), tf.float32)

        # calculate loss1, update opt1
        loss1 = (-1. * tf.reduce_sum(ce(self.net_pre1, self.net_pre2), reduction_indices=None) * self.not_valid) + \
                tf.reduce_sum(self.lambda_ * self.distance_mat * self.not_valid) + 1e-4 * tf.reduce_sum(
            tf.norm(self.z1, axis=-1) * self.not_valid) + \
                1e-4 * tf.reduce_sum(tf.norm(self.z2, axis=-1) * self.not_valid) / tf.reduce_sum(self.not_valid)

        self.opt_step1 = self.opt1.minimize(loss1, var_list=[self.z1, self.z2])
        loss2 = -1. * tf.reduce_mean(self.lambda_ * self.distance_mat * self.not_valid)
        self.opt_step2 = self.opt2.minimize(loss2, var_list=[self.lambda_])

    def perturb(self, sess, net, gan, eps, num_images=128,
             num_steps=1000):

        batch1 = tf.zeros((num_images, 3, 32, 32))
        batch2 = tf.zeros((num_images, 3, 32, 32))
        is_valid = tf.zeros(num_images)
        EPS = eps

        for i in range(num_images // BATCH_SIZE):
            # sample two latent code
            self.z1.assign(tf.random.uniform([BATCH_SIZE, embed_feats]))
            self.z2.assign(tf.random.uniform([BATCH_SIZE, embed_feats]))

            # reinitialize latent codes for new batch of images
            reset_optimizer_op = tf.variables_initializer([self.lambda_, self.opt1.variables(), self.opt2.variables()])
            sess.run(reset_optimizer_op)

            for j in range(num_steps):
                print("Steps: ", j)
                # generate images
                is_adv, is_feasible = sess.run([self.is_adv, self.is_feasible])
                if tf.reduce_sum(is_adv * is_feasible) == BATCH_SIZE:
                    batch1[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, ...] = sess.run(self.x1)
                    batch2[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, ...] = sess.run(self.x2)
                    is_valid[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = 1.
                    break

                # update latent code
                sess.run(self.opt_step1)
                sess.run(self.opt_step2)

                print(sess.run([-tf.reduce_sum(self.net_pre1 * tf.log(self.net_pre2), reduction_indices=[-1])]))
                print(sess.run(tf.reduce_sum(self.distance_mat)))

            batch1[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, ...] = sess.run(self.x1)
            batch2[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, ...] = sess.run(self.x2)
            is_valid[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = sess.run(self.is_adv) * sess.run(self.is_feasible)

        count = tf.reduce_sum(is_valid)
        print('number of adversarial pairs found:%d\n' % sess.run(count))

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


    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore the checkpoint
        saver.restore(sess, model_file)

        gan = hub.Module("https://tfhub.dev/google/compare_gan/model_13_cifar10_resnet_cifar/1")
        attack = L2OPAttack(gan, model)

        batch1, batch2, is_valid = attack.perturb(sess, model, gan, config['epsilon'])

        print('Storing examples')
        path = config['store_adv_path']
        batch1 = np.concatenate(batch1, axis=0)
        np.save(os.path.join(path, "batch1.npy"), batch1)

        batch2 = np.concatenate(batch2, axis=0)
        np.save(os.path.join(path, "batch2.npy"), batch2)
        print('Examples stored in {}'.format(path))
