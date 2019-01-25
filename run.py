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
from random import random, randint
from time import sleep
from optimizer.pytorch_optimizer import Vanilla_Unitary

def run(params):
    if params.bert_enabled == True:
        params.max_sequence_length = 512
        params.reader.max_sequence_length = 512
    model = models.setup(params)
    model = model.to(params.device)
    criterion = nn.CrossEntropyLoss()    
    
    optimizer_1 = None
    if params.network_type == 'mllm' or params.network_type == 'sentimllm':
        proj_measurements_params = list(model.proj_measurements.parameters())
        remaining_params =list(model.parameters())[:-7]+ list(model.parameters())[-7+len(proj_measurements_params):]
        optimizer = torch.optim.RMSprop(remaining_params, lr=0.001)
        if len(proj_measurements_params)>0:
            optimizer_1 = Vanilla_Unitary(proj_measurements_params,lr = 0.001, device = params.device)
    else:
        optimizer = torch.optim.RMSprop(list(model.parameters()), lr=0.001)
        
    output_file_path = 'output.txt'
    output_writer = open(output_file_path, 'w')
    
    t = trange(params.sample_num['train'])
    
    for i in range(params.epochs):
        print('epoch: ', i)
        train_accs = []
        losses = []
        
        for _i, sample_batched in enumerate(params.reader.get_train(iterable = True)):
            
            model.train()
            optimizer.zero_grad()
            if optimizer_1 is not None:
                optimizer_1.zero_grad()
            inputs = sample_batched['X'].to(params.device)
            targets = sample_batched['y'].to(params.device)
            if params.strategy == 'multi-task':
                senti_loss, outputs = model(inputs)
                loss = criterion(outputs, targets.argmax(1)) + params.gamma*senti_loss
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets.argmax(1))
            loss.backward()
            optimizer.step()
            if optimizer_1 is not None:
                optimizer_1.step()
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
        output_writer.write('epoch: {}\n'.format(i))
        output_writer.write('average train_acc: {}, average train_loss: {}\n'.format(avg_train_acc, avg_loss))
        
        test_losses = []
        n_correct = 0
        n_total = 0
        model.eval()
        for _i, sample_batched in enumerate(params.reader.get_test(iterable = True)):
            test_inputs = sample_batched['X'].to(params.device)
            test_targets = sample_batched['y'].to(params.device)
            with torch.no_grad():
                if params.strategy == 'multi-task':
                    senti_acc, test_outputs = model(test_inputs)
                else:
                    test_outputs = model(test_inputs)
                n_correct += (torch.argmax(test_outputs, -1) == torch.argmax(test_targets, -1)).sum().item()
                n_total += len(test_outputs)
                loss = criterion(test_outputs, torch.max(test_targets, 1)[1])
            test_losses.append(loss.item())
        test_acc = n_correct / n_total
        avg_test_loss = np.mean(test_losses)
        print('test_acc: {}, test_loss: {}\n\n\n'.format(test_acc, avg_test_loss.item()))
        output_writer.write('test_acc: {}, test_loss: {}\n\n\n'.format(test_acc, avg_test_loss.item()))
        output_writer.flush()
        


if __name__=="__main__":
    grid_parameters ={
        #"dataset_name":["SST_2"],
        "measurement_size" :[10,20,30],#,50100],
        "ngram_value": ["2","3","5","2,3","2,5","3,5","2,3,5"]
    }
#    parameters= [arg for index,arg in enumerate(itertools.product(*grid_parameters.values())) if index%args.gpu_num==args.gpu]
    parameters= [arg for index,arg in (enumerate(itertools.product(*grid_parameters.values())))]
    parameters= parameters[::-1]
    
    params = Params()
    params.seed = 9999
    if torch.cuda.is_available():
        params.device = torch.device('cuda')
        torch.cuda.manual_seed(params.seed)
    else:
        params.device = torch.device('cpu')
        torch.manual_seed(params.seed)
    config_file = 'config/config_multilayer.ini'    # define dataset in the config
    params.parse_config(config_file)  
    
    for parameter in parameters:
        print(parameter)
        params.setup(zip(grid_parameters.keys(),parameter))
        reader = dataset.setup(params)
        params.reader = reader
        
    #        if torch.cuda.is_available() else 'cpu')
        
        run(params)

