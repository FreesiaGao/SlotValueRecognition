# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf

import utils


class LSTMCRF:

    def __init__(self, config, initializer=tf.random_normal_initializer(stddev=0.1)):
        self.config = config
        self.initializer = initializer
        self.sess = None
        self.saver = None

    def add_placeholders(self):
        # shape = [batch size, max length of sentence in batch]
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name='word_ids')
        # shape = [batch size]
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        # shape = [batch size, max length of sentence in batch]]
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')

        self.dropout = tf.placeholder(tf.float32, shape=[], name='dropout')
        self.lr = tf.placeholder(tf.float32, shape=[], name='lr')

    def add_word_embeddings_op(self):
        # word embeddings op
        self.embedding_matrix = tf.get_variable(name='embedding_matrix', shape=[self.config.vocab_size, self.config.dim_word], initializer=self.initializer)
        self.word_embeddings = tf.nn.embedding_lookup(self.embedding_matrix, self.word_ids)

    def add_logits_op(self):
        # logits op
        cell_fw = tf.contrib.rnn.BasicLSTMCell(self.config.dim_hidden)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(self.config.dim_hidden)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, self.word_embeddings,
            sequence_length=self.sequence_lengths, dtype=tf.float32
        )
        # shape = [batch size, max length of sentence in batch, 2 * hidden size]
        output = tf.concat([output_fw, output_bw], axis=-1)
        W = tf.get_variable('W', shape=[2 * self.config.dim_hidden, self.config.label_size], dtype=tf.float32,
                            initializer=self.initializer)
        b = tf.get_variable('b', shape=[self.config.label_size], dtype=tf.float32, initializer=self.initializer)
        nsteps = tf.shape(output)[1]
        output = tf.reshape(output, [-1, 2 * self.config.dim_hidden])
        pred = tf.matmul(output, W) + b
        self.logits = tf.reshape(pred, [-1, nsteps, self.config.label_size])

    def add_loss_op(self):
        # loss op
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
            self.logits, self.labels, self.sequence_lengths
        )
        self.trans_params = trans_params
        self.loss = tf.reduce_mean(-log_likelihood)

    def add_train_op(self):
        # train op
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)

    def init_session(self):
        print('Initializeing tf session')
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def save_session(self):
        self.saver.save(self.sess, self.config.filename_model)

    def clos_session(self):
        self.sess.close()

    def build(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_loss_op()

        self.add_train_op()
        self.init_session()

    def get_feed_dict(self, batch_words, batch_labels=None, sequence_lengths=None, lr=None, dropout=None):
        words_id = []
        for words in batch_words:
            words = [w[0] for w in words]
            words_id.append(words)
        feed = {
            self.word_ids: words_id,
            self.sequence_lengths: sequence_lengths
        }

        if batch_labels is not None:
            feed[self.labels] = batch_labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed

    def predict(self, batch_words, sequence_lengths):
        fd = self.get_feed_dict(batch_words, sequence_lengths=sequence_lengths, dropout=1.0)
        logits, trans_params = self.sess.run([self.logits, self.trans_params], feed_dict=fd)

        pred_labels, pred_scores = [], []
        for logit, length in zip(logits, sequence_lengths):
            logit = logit[:length]
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            pred_labels.append(viterbi_seq)
            pred_scores.append(viterbi_score)

        return pred_labels, pred_scores

    def evaluate(self, test):
        """
        evaluates performance on test set
        :param test: dataset that yields tuple of (sentences, tags)
        :return: metrics: (dict) metrics['acc'] = 98.4, ...
        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.

        test_data, sequence_lengths = test.get_batch(test.size())
        test_words = [instance[0] for instance in test_data]
        test_labels = [instance[1] for instance in test_data]
        pred_labels, pred_scores = self.predict(test_words, sequence_lengths)

        for lab, lab_pred, length in zip(test_labels, pred_labels, sequence_lengths):
            lab = lab[:length]
            lab_pred = lab_pred[:length]
            accs += [a == b for (a, b) in zip(lab, lab_pred)]

            lab_chunks = set(utils.get_chunks(lab, self.config.vocab_labels))
            lab_pred_chunks = set(utils.get_chunks(lab_pred, self.config.vocab_labels))

            correct_preds += len(lab_chunks & lab_pred_chunks)
            total_preds += len(lab_pred_chunks)
            total_correct += len(lab_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        score = np.mean(pred_scores)
        return {'score': score, 'acc': acc, 'f1': f1, 'p': p, 'r': r}

    def train(self, train_set, evaluate_set):
        best_score = 0
        nepoch_no_imprv = 0     # for early stopping

        for epoch in range(self.config.nepoch):
            print('Epoch {:} out of {:}'.format(epoch+1, self.config.nepoch))

            nbatch = int((train_set.size()+self.config.batch_size-1) / self.config.batch_size)
            for i in range(nbatch):
                batch, sequence_lengths = train_set.get_batch(self.config.batch_size)
                batch_words = [instance[0] for instance in batch]
                batch_labels = [instance[1] for instance in batch]
                fd = self.get_feed_dict(batch_words, batch_labels, sequence_lengths, self.config.lr,
                                        self.config.dropout)
                _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict=fd)
                print('epoch {:} batch {:} : train_loss: {:}'.format(epoch, i, train_loss))

            metrics = self.evaluate(evaluate_set)
            print(str(metrics))

            score = metrics['score']
            self.config.lr *= self.config.lr_decay      # decay learning rate

            # early stopping and saving best parameters
            if score >= best_score:
                nepoch_no_imprv = 0
                self.save_session()
                best_score = score
                print('- new best score!')
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    print('- early stopping {} epochs without improvement'.format(nepoch_no_imprv))
                    break

