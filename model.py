# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf

import utils


class Model:

    def __init__(self, config, initializer=tf.random_normal_initializer(stddev=0.1)):
        self.config = config
        self.initializer = initializer
        self.sess = None
        self.saver = None

    def add_placeholders(self):
        # shape = [batch size, max length of sentence in batch]
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name='word_ids')
        self.ner_ids = tf.placeholder(tf.int32, shape=[None, None], name='ner_ids')
        self.position_w_ids = tf.placeholder(tf.int32, shape=[None, None], name='position_w_ids')
        self.position_s_ids = tf.placeholder(tf.int32, shape=[None, None], name='position_s_ids')
        # shape = [batch size]
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        # shape = [batch size, max length of sentence in batch]]
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')

        self.dropout = tf.placeholder(tf.float32, shape=[], name='dropout')
        self.lr = tf.placeholder(tf.float32, shape=[], name='lr')

    def add_word_embeddings_op(self):
        # word embeddings op
        self.embedding_matrix_word = tf.get_variable(name='embedding_matrix_word', shape=[self.config.vocab_size, self.config.dim_word], initializer=self.initializer)
        # shape = [batch size, sentence length, dim_word]
        self.word_embeddings = tf.nn.embedding_lookup(self.embedding_matrix_word, self.word_ids)
        self.embedding_matrix_ner = tf.get_variable(name='embedding_matrix_ner', shape=[self.config.ner_size, self.config.dim_ner], initializer=self.initializer)
        # shape = [batch size, dim_ner]
        self.ner_embeddings = tf.nn.embedding_lookup(self.embedding_matrix_ner, self.ner_ids)
        # shape = [batch size, dim_word + dim_ner]
        self.inputs_embeddings = tf.concat([self.word_embeddings, self.ner_embeddings], axis=-1)
        self.embedding_matrix_position_w = tf.get_variable(name='embedding_matrix_position_w', shape=[self.config.position_w_size, self.config.dim_position_w], initializer=self.initializer)
        self.position_w_embeddings = tf.nn.embedding_lookup(self.embedding_matrix_position_w, self.position_w_ids)
        self.embedding_matrix_position_s = tf.get_variable(name='embedding_matrix_position_s', shape=[self.config.position_s_size, self.config.dim_position_s], initializer=self.initializer)
        # shape = [batch size, sentence length, dim_position_s]
        self.position_s_embeddings = tf.nn.embedding_lookup(self.embedding_matrix_position_s, self.position_s_ids)

    def add_logits_op(self):
        # logits op
        cell_fw = tf.contrib.rnn.BasicLSTMCell(self.config.dim_hidden)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(self.config.dim_hidden)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, self.inputs_embeddings,
            sequence_length=self.sequence_lengths, dtype=tf.float32
        )
        # shape = [batch size, sentence length, 2 * hidden size]
        self.lstm_output = tf.concat([output_fw, output_bw], axis=-1)
        W_logit = tf.get_variable('W_logit', shape=[2 * self.config.dim_hidden, self.config.label_size],
                                  dtype=tf.float32, initializer=self.initializer)
        b_logit = tf.get_variable('b_logit', shape=[self.config.label_size], dtype=tf.float32,
                                  initializer=self.initializer)
        nsteps = tf.shape(self.lstm_output)[1]
        # shape = [batch size * sentence length, label_size]
        output = tf.reshape(self.lstm_output, [-1, 2 * self.config.dim_hidden])
        pred = tf.matmul(output, W_logit) + b_logit
        # shape = [batch size, sentence length, label_size]
        self.logits = tf.reshape(pred, [-1, nsteps, self.config.label_size])

    def add_attention_op(self):
        if self.config.use_attention:
            nstep = tf.shape(self.logits)[1]
            hidden = tf.reshape(self.lstm_output, [-1, 2 * self.config.dim_hidden])
            W_hidden = tf.get_variable('W_hideen', shape=[2 * self.config.dim_hidden, self.config.dim_attention],
                                       dtype=tf.float32, initializer=self.initializer)
            # shape = [batch size * text length, dim_attention]
            Wh = tf.matmul(hidden, W_hidden)
            Wh = tf.reshape(Wh, [-1, nstep, self.config.dim_attention])

            W_position = tf.get_variable('W_position', shape=[self.config.dim_position_w, self.config.dim_attention],
                                         dtype=tf.float32, initializer=self.initializer)
            # shape = [batch size * text length, dim_attention]
            position = tf.reshape(self.position_w_embeddings, [-1, self.config.dim_position_w])
            Wp = tf.matmul(position, W_position)
            Wp = tf.reshape(Wp, [-1, nstep, self.config.dim_attention])

            Wh = tf.reduce_sum(Wh, axis=0)
            Wp = tf.reduce_sum(Wp, axis=0)
            W_attention = tf.get_variable(name='W_attention', shape=[self.config.dim_attention, 1], dtype=tf.float32,
                                          initializer=self.initializer)
            # shape = [nstep, 1]
            h = tf.matmul(Wh, W_attention)
            p = tf.matmul(Wp, W_attention)
            # shape = [1, nstep]
            ht = tf.reshape(h, [1, -1])
            h = tf.tile(h, [1, nstep])
            ht = tf.tile(ht, [nstep, 1])
            attention = h + p + ht

            # shape = [nstep, nstep]
            self.attention = tf.nn.softmax(attention)
            # shape = [batch size * nstep, label_size]
            batch = tf.shape(self.logits)[0]
            # shape = [batch size, sentence length, sentence length]
            attention = tf.tile(tf.reshape(self.attention, [-1, nstep, nstep]), [batch, 1, 1])
            self.logits = tf.matmul(attention, self.logits)

    def add_factor_op(self):
        if self.config.use_factor:
            W_factor = tf.get_variable('W_factor', shape=[self.config.dim_position_s, self.config.label_size],
                                       dtype=tf.float32, initializer=self.initializer)
            nsteps = tf.shape(self.position_s_embeddings)[1]
            # shape = [batch size * sentence length, dim_position_s]
            position_embedding = tf.reshape(self.position_s_embeddings, [-1, self.config.dim_position_s])
            # shape = [batch size * sentence length, label_size]
            factor = tf.matmul(position_embedding, W_factor)
            factor = tf.reshape(factor, [-1, nsteps, self.config.label_size])
            self.logits = tf.multiply(self.logits, factor)

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

    def save_session(self, metrics, epoch):
        self.saver.save(self.sess, self.config.filename_model)
        with open(self.config.filename_model+'_metrics', 'w') as writer:
            writer.write('stopped in ' + str(epoch) + 'th epoch\n')
            writer.write(str(metrics))

    def clos_session(self):
        self.sess.close()

    def build(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_attention_op()
        self.add_factor_op()
        self.add_loss_op()

        self.add_train_op()
        self.init_session()

    def get_feed_dict(self, batch_words, batch_labels=None, sequence_lengths=None, lr=None, dropout=None):
        word_ids, ner_ids, position_w_ids, position_s_ids = [], [], [], []
        for senuence in batch_words:
            words = [w[0] for w in senuence]
            ners = [w[1] for w in senuence]
            position_ws = [w[2] for w in senuence]
            position_ss = [w[3] for w in senuence]
            word_ids.append(words)
            ner_ids.append(ners)
            position_w_ids.append(position_ws)
            position_s_ids.append(position_ss)
        feed = {
            self.word_ids: word_ids,
            self.ner_ids: ner_ids,
            self.position_w_ids: position_w_ids,
            self.position_s_ids: position_s_ids,
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

        p = correct_preds / total_preds if total_preds > 0 else 0
        r = correct_preds / total_correct if total_correct > 0 else 0
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
                self.save_session(metrics, epoch)
                best_score = score
                print('- new best score!')
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    print('- early stopping {} epochs without improvement'.format(nepoch_no_imprv))
                    break

