# -*- coding: utf-8 -*-
""" main.py """

from configs.config import CFG
from model.cnn_lstm import CNN_LSTM


def run():
    """Builds model, loads data, trains and evaluates"""
    model = CNN_LSTM(CFG)
    model.load_data()
    model.build()
#    model.load_saved_model()
#    model.train()
    model.evaluate()


if __name__ == '__main__':
    run()