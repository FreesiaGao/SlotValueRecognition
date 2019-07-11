# -*- coding:utf-8 -*-

import tensorflow as tf

import config  as cfg
from data_builder import DataBuilder
from model import Model
from lstm_crf import LSTMCRF


def main(_):
    config = cfg.Config()
    data_builder = DataBuilder(config)
    train_set, evaluate_set = data_builder.build_data(config.path_data)

    model = Model(config)
    model.build()
    model.train(train_set, evaluate_set)


if __name__ == '__main__':
    tf.app.run()




