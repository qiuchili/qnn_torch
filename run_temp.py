# -*- coding: utf-8 -*-

from params import Params
import dataset
import numpy as np
import torch
import torch.nn as nn
import models
from optimizer.pytorch_optimizer import Vanilla_Unitary

def run(params):
    model = models.setup(params)
    model = model.to(params.device)
    criterion = nn.CrossEntropyLoss()    
    
    optimizer_1 = None
    if params.network_type == 'mllm':
        proj_measurements_params = list(model.proj_measurements.parameters())
        remaining_params =list(model.parameters())[:-6]+ list(model.parameters())[-6+len(proj_measurements_params):]
        optimizer = torch.optim.RMSprop(remaining_params, lr=params.lr)
        if len(proj_measurements_params)>0:
            optimizer_1 = Vanilla_Unitary(proj_measurements_params,lr=params.lr, device = params.device)
    else:
        optimizer = torch.optim.RMSprop(list(model.parameters()), lr=params.lr)

    max_test_acc = 0.
    for i in range(params.epochs):
        print('epoch: ', i)
        train_accs = []
        train_losses = []
        for _i, sample_batched in enumerate(params.reader.get_train(iterable = True)):
            model.train()
            optimizer.zero_grad()
            inputs = sample_batched['X'].to(params.device)
            targets = sample_batched['y'].to(params.device)
            if params.strategy == 'multi-task':
                senti_outputs, senti_targets, outputs = model(inputs)
                loss = criterion(outputs, targets.argmax(1)) + params.gamma*criterion(senti_outputs, senti_targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets.argmax(1))
            loss.backward()
            optimizer.step()
            if optimizer_1 is not None:
                optimizer_1.step()
            n_correct = (outputs.argmax(1) == targets.argmax(1)).sum().item()
            n_total = len(outputs)
            train_acc = n_correct / n_total
            train_accs.append(train_acc)
            train_losses.append(loss.item())

            if _i % 50 == 0:
                model.eval()
                avg_train_acc = np.mean(train_accs)
                avg_train_loss = np.mean(train_losses)
                t_n_correct = 0
                t_n_total = 0
                for t_sample_batched in params.reader.get_test(iterable = True):
                    t_inputs = t_sample_batched['X'].to(params.device)
                    t_targets = t_sample_batched['y'].to(params.device)
                    with torch.no_grad():
                        if params.strategy == 'multi-task':
                            _, _, t_outputs = model(t_inputs)
                        else:
                            t_outputs = model(t_inputs)
                        t_n_correct += (t_outputs.argmax(1) == t_targets.argmax(1)).sum().item()
                        t_n_total += len(t_outputs)
                test_acc = t_n_correct / t_n_total
                if test_acc > max_test_acc:
                    max_test_acc = test_acc
                print('average_train_acc: {}, average_train_loss: {}, test_acc: {}'.format(avg_train_acc, avg_train_loss, test_acc))
    
    print('max_test_acc: {}'.format(max_test_acc))



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

