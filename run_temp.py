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
from tqdm import tqdm,trange

def run(params):
    model = models.setup(params)
    model = model.to(params.device)
    criterion = nn.CrossEntropyLoss()    
    
    optimizer = torch.optim.RMSprop(model.parameters(), lr=params.lr)
    
    t = trange(params.sample_num['train'])
    
    max_test_acc = 0.
    for i in range(params.epochs):
        print('epoch: ', i)
        train_accs = []
        losses = []
        
        
        for _i, sample_batched in enumerate(params.reader.get_train(iterable = True)):
            
            model.train()
            optimizer.zero_grad()
            inputs = sample_batched['X'].to(params.device)
            targets = sample_batched['y'].to(params.device)
            if params.strategy == 'multi-task':
                senti_outputs, senti_targets, outputs = model(inputs)
                loss = criterion(outputs, torch.max(targets, 1)[1]) + params.gamma*criterion(senti_outputs, senti_targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, torch.max(targets, 1)[1])
            loss.backward()
            optimizer.step()
            n_correct = (torch.argmax(outputs, -1) == torch.argmax(targets, -1)).sum().item()
            n_total = len(outputs)
            train_acc = n_correct / n_total
            t.update(params.batch_size)
            t.set_postfix(loss=loss.item(), train_acc=train_acc)
            train_accs.append(train_acc)
            losses.append(loss.item())
        
        t.clear()
        
        avg_train_acc = np.mean(train_accs)
        avg_loss = np.mean(losses)
        print('average train_acc: {}, average train_loss: {}'.format(avg_train_acc, avg_loss))
        
        test_losses = []
        n_correct = 0
        n_total = 0
        for _i, sample_batched in enumerate(params.reader.get_test(iterable = True)):
            test_inputs = sample_batched['X'].to(params.device)
            test_targets = sample_batched['y'].to(params.device)
            with torch.no_grad():
                test_outputs = model(test_inputs.long())
                n_correct += (torch.argmax(test_outputs, -1) == torch.argmax(test_targets, -1)).sum().item()
                n_total += len(test_outputs)
                loss = criterion(test_outputs, torch.max(test_targets, 1)[1])
            test_losses.append(loss.item())
        test_acc = n_correct / n_total
        if test_acc > max_test_acc:
            max_test_acc = test_acc
        avg_test_loss = np.mean(test_losses)
        print('test_acc: {}, test_loss: {}\n\n\n'.format(test_acc, avg_test_loss.item()))
    
    print('max test_acc: {}'.format(max_test_acc))



if __name__=="__main__":
  
    params = Params()
    config_file = 'config/config_sentimllm.ini'    # define dataset in the config
    params.parse_config(config_file)    
    
    reader = dataset.setup(params)
    params.reader = reader
    
    if torch.cuda.is_available():
        params.device = torch.device('cuda')
        torch.cuda.manual_seed(params.seed)
    else:
        params.device = torch.device('cpu')
        torch.manual_seed(params.seed)

    run(params)

