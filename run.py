# -*- coding: utf-8 -*-

from params import Params
import dataset
from units import to_array, batch_softmax_with_first_item
import itertools
import argparse
import numpy as np
import preprocess.embedding
import torch
import torch.nn as nn
import models
from optimizer.pytorch_optimizer import Vanilla_Unitary

def run(params):
    if params.bert_enabled == True:
        params.max_sequence_length = 512
        params.reader.max_sequence_length = 512
    model = models.setup(params)
    model = model.to(params.device)
    criterion = nn.CrossEntropyLoss()    
    proj_measurements_params = list(model.proj_measurements.parameters())
    print(len(proj_measurements_params))
    remaining_params =list(model.parameters())[:-7]+ list(model.parameters())[-7+len(proj_measurements_params):]
    optimizer = torch.optim.RMSprop(remaining_params, lr=0.01)
    optimizer_1 = Vanilla_Unitary(proj_measurements_params,lr = 0.01, device = params.device)

    test_x, test_y = params.reader.get_test(iterable = False)
    test_inputs = torch.tensor(test_x).to(params.device)
    test_targets = torch.tensor(test_y).to(params.device)
    
    for i in range(params.epochs):
        print('epoch: ', i)
        train_accs = []
        losses = []
        for _i,sample_batched in enumerate(params.reader.get_train(iterable = True)):
            model.train()
            optimizer.zero_grad()
            optimizer_1.zero_grad()
            inputs = sample_batched['X'].to(params.device)
            targets = sample_batched['y'].to(params.device)
            outputs = model(inputs)
            loss = criterion(outputs, torch.max(targets, 1)[1])
            loss.backward()
            optimizer.step()
#            print('Updating Projection Layers:')
            optimizer_1.step()
            

            n_correct = (torch.argmax(outputs, -1) == torch.argmax(targets, -1)).sum().item()
            n_total = len(outputs)
            train_acc = n_correct / n_total
            if _i %100 ==0:
                print('train_acc: {}, loss: {}'.format(train_acc,loss.item()))
            train_accs.append(train_acc)
            losses.append(loss.item())
            
        avg_train_acc = np.mean(train_accs)
        avg_loss = np.mean(losses)
        print('average train_acc: {}, average train_loss: {}'.format(avg_train_acc, avg_loss))
        with torch.no_grad():
            test_outputs = model(test_inputs.long())
        n_correct = (torch.argmax(test_outputs, -1) == torch.argmax(test_targets, -1)).sum().item()
        n_total = len(test_outputs)
        test_acc = n_correct / n_total
        loss = criterion(test_outputs, torch.max(test_targets, 1)[1])
        print('test_acc: {}, test_loss: {}'.format(test_acc,loss.item()))



if __name__=="__main__":
  
    params = Params()
    config_file = 'config/config_multilayer.ini'    # define dataset in the config
    params.parse_config(config_file)    
    
    reader = dataset.setup(params)
    params.reader = reader
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    run(params)

