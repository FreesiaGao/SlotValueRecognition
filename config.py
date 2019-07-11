# -*- coding:utf-8 -*-

import os


class Config:

    path_data = 'data/labeled/'
    filename_labels = 'data/vocab/labels'
    filename_words = 'data/vocab/words'
    filename_ners = 'data/vocab/ners'
    filename_position_w = 'data/vocab/position_w'
    filename_position_s = 'data/vocab/position_s'

    feature_nums = 4
    PAD_TOKEN = 'PAD_TOKEN'
    PAD_WORD = (0, ) * feature_nums
    PAD_LABEL = 0
    default_label = 'O'

    dim_word = 32
    dim_ner = 5
    dim_position_w = 5
    dim_position_s = 5
    dim_hidden = 32
    dim_attention = 5

    batch_size = 5

    nepoch = 10  # epochs for training
    lr = 0.001
    lr_decay = 0.9
    dropout = 0.5
    nepoch_no_imprv = 3  # in recent n epochs has no improvement

    dir_model = ''

    use_attention = False
    use_factor = False

    def __init__(self):
        if not os.path.exists(self.filename_words) or not os.path.exists(self.filename_labels):
            self.build_vocab(self.path_data, self.filename_words, self.filename_ners,
                             self.filename_position_w, self.filename_position_s, self.filename_labels)
        self.vocab_labels, self.id_labels = self.load_vocab(self.filename_labels)
        self.label_size = len(self.vocab_labels)
        self.vocab_words, self.id_words = self.load_vocab(self.filename_words)
        self.vocab_size = len(self.vocab_words)
        self.vocab_ners, self.id_ners = self.load_vocab(self.filename_ners)
        self.ner_size = len(self.vocab_ners)
        self.vocab_position_w, self.id_position_w = self.load_vocab(self.filename_position_w)
        self.position_w_size = len(self.vocab_position_w)
        self.vocab_position_s, self.id_position_s = self.load_vocab(self.filename_position_s)
        self.position_s_size = len(self.vocab_position_s)

        self.filename_model = self.gen_filename()

    def gen_filename(self):
        return 'model/w' + str(self.dim_word) + 'n' + str(self.dim_ner) + 'p' + str(self.dim_position_w) + \
               'h' + str(self.dim_hidden) + 'a' + str(self.dim_attention) + '_batch' + str(self.batch_size) + \
               '_epoch' + str(self.nepoch) + '_lr' + str(self.lr) + '_dropout' + str(self.dropout) + '/model'

    def build_vocab(self, path_data, filename_words, filename_ners, filename_position_w, filename_position_s, filename_labels):
        """
        :param path_data:
        :param filename_words:
        :param filename_labels:
        :return:
        """
        words = set()
        ners = set()
        position_words = set()
        position_sentences = set()
        labels = set()

        flist = os.listdir(path_data)
        for file in flist:
            if file.startswith('.'):
                continue
            with open(path_data + file, 'r') as reader:
                for line in reader:
                    line = line[:-1].split('\t:\t')
                    words.add(line[0])
                    ners.add(line[1])
                    position_words.add(int(line[2]))
                    position_sentences.add(int(line[3]))
                    labels.add(line[4])

        word_list = list(words)
        word_list.sort()
        with open(filename_words, 'w') as writer:
            for word in word_list:
                writer.write(word + '\n')

        ner_list = list(ners)
        ner_list.sort()
        with open(filename_ners, 'w') as writer:
            for ner in ner_list:
                writer.write(ner + '\n')

        pw_list = list(position_words)
        pw_list.sort()
        with open(filename_position_w, 'w') as writer:
            for pw in pw_list:
                writer.write(str(pw) + '\n')

        ps_list = list(position_sentences)
        ps_list.sort()
        with open(filename_position_s, 'w') as writer:
            for ps in ps_list:
                writer.write(str(ps) + '\n')

        label_list = list(labels)
        label_list.sort()
        with open(filename_labels, 'w') as writer:
            for label in label_list:
                writer.write(label + '\n')

    def load_vocab(self, filename):
        """
        :param filename:
        :return:
        """
        vocab_id = {self.PAD_TOKEN: self.PAD_LABEL}
        id_vocab = {self.PAD_LABEL: self.PAD_TOKEN}
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word[:-1]
                vocab_id[word] = idx+1
                id_vocab[idx+1] = word
        return vocab_id, id_vocab
