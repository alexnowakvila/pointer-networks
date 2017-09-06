"""Implementation of Pointer networks: http://arxiv.org/pdf/1506.03134v1.pdf.
"""


from __future__ import absolute_import, division, print_function

import random
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf

from dataset import DataGenerator
from pointer import pointer_decoder

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 128, 'Batch size.  ')
flags.DEFINE_integer('max_len', 50, 'Size of problem.')
flags.DEFINE_integer('num_steps', 100000, 'Number of steps to train for')
flags.DEFINE_integer('rnn_size', 512, 'Number of RNN cells in each layer')
flags.DEFINE_integer('num_layers', 1, 'Number of layers in the network.')
flags.DEFINE_integer('load_from_checkpoint', False, 'Whether to load from checkpoint')
flags.DEFINE_string('checkpoint_dir', 'checkpoints', 'Directory to store checkpoints')
flags.DEFINE_string('log_dir', 'pointer_logs', 'Directory to put tensorboard log files')
flags.DEFINE_string('problem_type', 'convex_hull', 'What kind of problem to train on: "convex_hull", or "sort".')
flags.DEFINE_string('pointer_type', 'one_hot', 'What kind of pointer to use: "multi_hot", "one_hot", or "soft_max"')
flags.DEFINE_integer('steps_per_checkpoint', 200, 'How many training steps to do per checkpoint.')
flags.DEFINE_float('learning_rate', 0.001, "Learning rate.")
flags.DEFINE_boolean('to_csv', True, "if true, export the averaged loss and test accuracies")
FLAGS = flags.FLAGS

class PointerNetwork(object):
    
    def __init__(self, max_len, input_size, size, num_layers, batch_size, learning_rate):
        """Create the network.
        
        Args:
            max_len: maximum length of the model.
            input_size: size of the inputs data.
            size: number of units in each layer of the model.
            num_layers: number of layers in the model.
            batch_size: the size of the batches used during training;
                the model construction is independent of batch_size, so it can be
                changed after initialization if this is convenient, e.g., for decoding.
            learning_rate: learning rate to start with.
        """
        self.max_len = max_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, trainable=False)

        
        # cell = tf.contrib.rnn.GRUCell(size, initializer=tf.random_uniform_initializer(-0.08, 0.08))
        stacked_rnn_enc = []
        for iiLyr in range(num_layers):
            # stacked_rnn_enc.append(tf.nn.rnn_cell.GRUCell(size))
            stacked_rnn_enc.append(tf.nn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True))
        cell_enc = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_enc)

        stacked_rnn_dec = []
        for iiLyr in range(num_layers):
            # stacked_rnn_dec.append(tf.nn.rnn_cell.GRUCell(size))
            stacked_rnn_dec.append(tf.nn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True))
        cell_dec = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_dec)
        # if num_layers > 1:
        #     cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
            
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.decoder_targets = []
        self.target_weights = []
        for i in range(max_len):
            self.encoder_inputs.append(tf.placeholder(
                tf.float32, [batch_size, input_size], name="EncoderInput%d" % i))

        for i in range(max_len + 1):
            self.decoder_inputs.append(tf.placeholder(
                tf.float32, [batch_size, input_size], name="DecoderInput%d" % i))
            self.decoder_targets.append(tf.placeholder(
                tf.float32, [batch_size, max_len + 1], name="DecoderTarget%d" % i))  # one hot
            self.target_weights.append(tf.placeholder(
                tf.float32, [batch_size, 1], name="TargetWeight%d" % i))
        
        # Need for attention
        encoder_outputs, final_state = tf.contrib.rnn.static_rnn(cell_enc, self.encoder_inputs, dtype = tf.float32)
        
        # Need a dummy output to point on it. End of decoding.
        encoder_outputs = [tf.zeros([batch_size, size])] + encoder_outputs

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [tf.reshape(e, [-1, 1, cell_enc.output_size])
                      for e in encoder_outputs]
        attention_states = tf.concat(axis=1, values=top_states)

        #For training
        with tf.variable_scope("decoder"):
            outputs, states, _ = pointer_decoder(
                self.decoder_inputs, final_state, attention_states, cell_dec, feed_prev=False, pointer_type=FLAGS.pointer_type)

        #For inference
        with tf.variable_scope("decoder", reuse=True):
            predictions, _, inps = pointer_decoder(
                self.decoder_inputs, final_state, attention_states, cell_dec, feed_prev=True, pointer_type=FLAGS.pointer_type)
            
        self.predictions = predictions
        self.outputs = outputs
        self.inps = inps
        
            
    def create_feed_dict(self, encoder_input_data, decoder_input_data, decoder_target_data):
        feed_dict = {}
        for placeholder, data in zip(self.encoder_inputs, encoder_input_data):
            feed_dict[placeholder] = data

        for placeholder, data in zip(self.decoder_inputs, decoder_input_data):
            feed_dict[placeholder] = data

        for placeholder, data in zip(self.decoder_targets, decoder_target_data):
            feed_dict[placeholder] = data

        for placeholder in self.target_weights:
            feed_dict[placeholder] = np.ones([self.batch_size, 1])

        return feed_dict

    def step(self):

        loss = 0.0
        for output, target, weight in zip(self.outputs, self.decoder_targets, self.target_weights):
            loss += tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=target) * weight

        loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss', loss)
        
        test_loss = 0.0
        for output, target, weight in zip(self.predictions, self.decoder_targets, self.target_weights):
            test_loss += tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=target) * weight
        
        tf.summary.histogram('predictions', self.predictions)
        test_loss = tf.reduce_mean(test_loss)
        tf.summary.scalar('test_loss', test_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss)
        
        train_loss_value = 0.0
        test_loss_value = 0.0
        test_acc_value = 0.0
        
        correct_order = 0.0
        all_order = 0.0

        predictions_order = tf.concat(axis=0, values=[tf.expand_dims(prediction , 0) for prediction in self.predictions])
        predictions_order = tf.transpose(tf.argmax(predictions_order, 2), perm=[1,0])
        
        targets_order = tf.concat(axis=0, values=[tf.expand_dims(target, 0) for target in self.decoder_targets])

        targets_order = tf.transpose(tf.argmax(targets_order, 2), perm=[1,0])
        
        correct_order += tf.reduce_sum(tf.cast(tf.reduce_all(tf.equal(predictions_order,targets_order), 1), tf.float32))
        all_order += self.batch_size

        acc = correct_order/all_order
        tf.summary.scalar('accuracy', acc)

        sess = tf.Session()
        previous_losses = []
        test_losses = []
        test_accuracies = []
        merged = tf.summary.merge_all()

        #add op to save and restore all the variables
        saver = tf.train.Saver()

        with sess.as_default():
            train_writer = tf.summary.FileWriter(FLAGS.log_dir + "/" + FLAGS.problem_type +"/" + FLAGS.pointer_type+ "/train", sess.graph)
            test_writer = tf.summary.FileWriter(FLAGS.log_dir + "/" + FLAGS.problem_type +"/" + FLAGS.pointer_type + "/test", sess.graph)
            init = tf.global_variables_initializer()
            sess.run(init)

            if FLAGS.load_from_checkpoint:
                print("Loading from checkpoint...")
                saver.restore(sess, FLAGS.checkpoint_dir+"/" + FLAGS.pointer_type +  "/model.ckpt")
            print("Training network...")

            for i in range(FLAGS.num_steps): 
                encoder_input_data, decoder_input_data, targets_data = dataset.next_batch(
                    self.batch_size, self.max_len, convex_hull=(FLAGS.problem_type=="convex_hull"))
                # Train
                feed_dict = self.create_feed_dict(
                    encoder_input_data, decoder_input_data, targets_data)

                if (i+1)%FLAGS.steps_per_checkpoint == 0:
                    #record run metadata
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, d_x, l = sess.run([merged, loss, train_op], 
                        feed_dict=feed_dict, 
                        options=run_options,
                        run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%d'%(i+1))
                    train_writer.add_summary(summary, (i+1))
                else:
                    summary, d_x, l = sess.run([merged, loss, train_op], feed_dict=feed_dict)
                    train_writer.add_summary(summary, (i+1))

                train_loss_value += d_x/FLAGS.steps_per_checkpoint

                if (i+1) % FLAGS.steps_per_checkpoint == 0:
                    print('Step:', i+1, 'Learning rate:', self.learning_rate)
                    # store checkpoint
                    saver.save(sess, FLAGS.checkpoint_dir+"/" + FLAGS.pointer_type +  "/model.ckpt")
                    print("Train Loss: ", train_loss_value)
                    previous_losses.append(train_loss_value)
                    train_loss_value = 0

                encoder_input_data, decoder_input_data, targets_data = dataset.next_batch(
                    self.batch_size, self.max_len, train_mode=False, convex_hull=(FLAGS.problem_type=="convex_hull"))
                # Test
                feed_dict = self.create_feed_dict(
                    encoder_input_data, decoder_input_data, targets_data)
                inps_ = sess.run(self.inps, feed_dict=feed_dict)
                predictions = sess.run(self.predictions, feed_dict=feed_dict)
                
                summary, test_loss_, test_acc = sess.run([merged, test_loss, acc], feed_dict=feed_dict)
                test_writer.add_summary(summary, (i+1))

                test_loss_value += test_loss_/FLAGS.steps_per_checkpoint
                test_acc_value += test_acc/FLAGS.steps_per_checkpoint

                if (i+1) % FLAGS.steps_per_checkpoint == 0:
                    print("Test Loss: ", test_loss_value)
                    test_losses.append(test_loss_value)
                    test_loss_value = 0.0
                    print('Test Accuracy: %.5f' % test_acc_value)
                    test_accuracies.append(test_acc_value)
                    test_acc_value = 0.0
                    print("----")
            # export data to csv
            if (FLAGS.to_csv):
                output=pd.DataFrame(data={'train_loss': previous_losses, 'test_loss': test_losses, 'test_accuracy': test_accuracies})
                output.to_csv('./pointer_logs/'+ FLAGS.problem_type+'_' + FLAGS.pointer_type+'.csv')


if __name__ == "__main__":
    # Make log and checkpoint directories if necessary
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    print("Creating pointer network...")
    pointer_network = PointerNetwork(FLAGS.max_len, 2 - (FLAGS.problem_type == 'sort'), FLAGS.rnn_size,
                                     FLAGS.num_layers, FLAGS.batch_size, FLAGS.learning_rate)
    dataset = DataGenerator()
    pointer_network.step()

