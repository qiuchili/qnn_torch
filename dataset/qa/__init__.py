# -*- coding: utf-8 -*-

from dataset.qa.QAHelper import dataHelper
from dataset.qa.data_reader import DataReader

def setup(opt):
    reader = DataReader(opt)# DataReader(opt)
    return reader
