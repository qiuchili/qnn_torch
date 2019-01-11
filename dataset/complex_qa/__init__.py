# -*- coding: utf-8 -*-

from dataset.complex_qa.QAHelper import dataHelper
from dataset.complex_qa.data_reader import DataReader

def setup(opt):
    reader = DataReader(opt)# DataReader(opt)
    return reader

