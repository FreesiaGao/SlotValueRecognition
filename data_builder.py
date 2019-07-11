# -*- coding:utf-8 -*-

import os
import random

import config as cfg


class DataSet:
    def __init__(self, data, curr=0):
        self.data = data        # [(words, labels), (words, labels), (words, labels)]
        self.curr = curr

    def size(self):
        return len(self.data)

    def get_batch(self, batch_size):
        end = self.curr + batch_size
        batch, seq_lengths = self.pad(self.data[self.curr:end])
        if end > len(self.data):
            self.curr = 0
        return batch, seq_lengths

    def pad(self, batch):
        batch_words = [instance[0] for instance in batch]
        batch_labels = [instance[1] for instance in batch]
        max_length = max(map(lambda x: len(x), batch_words))
        batch_padded, seq_length = [], []
        for words, labels in zip(batch_words, batch_labels):
            length = len(words)
            words_ = words[:max_length] + [cfg.Config.PAD_WORD] * max(max_length - length, 0)
            labels_ = labels[:max_length] + [cfg.Config.PAD_LABEL] * max(max_length - length, 0)
            batch_padded.append((words_, labels_))
            seq_length.append(length)
        return batch_padded, seq_length


class DataBuilder:

    def __init__(self, config):
        self.config = config

    def load_instance(self, filename):
        """
        translate a text file to an instance
        :param filename: format: word \t:\t label
        :return: instance, format: [(word_id, position_word, position_sentence)], [label_id]
        """
        words = []
        labels = []
        with open(filename, 'r') as reader:
            for line in reader:
                line = line[:-1].split('\t:\t')
                word_id = self.config.vocab_words[line[0]]
                ner_id = self.config.vocab_ners[line[1]]
                position_word = self.config.vocab_position_w[line[2]]
                position_sentence = self.config.vocab_position_s[line[3]]
                words.append((word_id, ner_id, position_word, position_sentence))
                if len(line) > self.config.feature_nums:
                    label_id = self.config.vocab_labels[line[self.config.feature_nums]]
                    labels.append(label_id)
        return words, labels

    def build_data(self, path_data, train_radio=0.7):
        """
        build train and eva
        :param path_data:
        :param train_radio: train set radio
        :return: train_set, evaluate_set
        """
        data = []
        flist = os.listdir(path_data)
        for file in flist:
            if file.startswith('.'):
                continue
            words, labels = self.load_instance(path_data+file)
            data.append((words, labels))
        random.shuffle(data)
        size = len(data)
        return DataSet(data[:int(train_radio*size)]), DataSet(data[int(train_radio*size)+1:])

    def instance2text(self, words=None, labels=None):
        """
        translate an instance to text
        :param words:
        :param labels:
        :return:
        """
        if words:
            words = [self.config.id_words[w] for w in words]
        if labels:
            labels = [self.config.id_labels[w] for w in labels]
        return words, labels


if __name__ == '__main__':
    config = cfg.Config()
    dataBuilder = DataBuilder(config)
    train_set, dev_set = dataBuilder.build_data(config.path_data)
    batch, sequence_lengths = train_set.get_batch(config.batch_size)
    length = sequence_lengths[0]
    features = batch[0][0][:length]
    labels = batch[0][1][:length]
    print(length)
    words = [f[0] for f in features]
    ners = [f[1] for f in features]
    position_word = [f[2] for f in features]
    position_sentence = [f[3] for f in features]
    print(words)
    print(ners)
    print(position_word)
    print(position_sentence)
    print(labels)
    words, labels = dataBuilder.instance2text(words, labels)
    ners = [config.id_ners[w] for w in ners]
    print(words)
    print(ners)
    print(labels)

