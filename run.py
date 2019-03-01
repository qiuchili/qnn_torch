# -*- coding: utf-8 -*-

from params import Params
import dataset
import numpy as np
import torch
import torch.nn as nn
import models

def run(params):
    model = models.setup(params)
    model = model.to(params.device)
    criterion = nn.CrossEntropyLoss()
    senti_criterion = nn.CrossEntropyLoss() 
    
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
                senti_out, senti_tag, outputs = model(inputs)
                loss = criterion(outputs, targets.argmax(1)) + params.gamma*senti_criterion(senti_out, senti_tag)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets.argmax(1))
            loss.backward()
            optimizer.step()
            n_correct = (outputs.argmax(1) == targets.argmax(1)).sum().item()
            n_total = len(outputs)
            train_acc = n_correct / n_total
            train_accs.append(train_acc)
            train_losses.append(loss.item())

            if _i % 50 == 0:
                model.eval()
                avg_train_acc = np.mean(train_accs)
                avg_train_loss = np.mean(train_losses)
                train_accs = []
                train_losses = []
                t_n_correct = 0
                t_n_total = 0
                senti_acc = 0
                cnt = 0
                for t_sample_batched in params.reader.get_test(iterable = True):
                    t_inputs = t_sample_batched['X'].to(params.device)
                    t_targets = t_sample_batched['y'].to(params.device)
                    cnt += 1
                    with torch.no_grad():
                        if params.strategy == 'multi-task':
                            senti_acc_, t_outputs = model(t_inputs)
                            senti_acc += senti_acc_.item()
                        else:
                            t_outputs = model(t_inputs)
                        t_n_correct += (t_outputs.argmax(1) == t_targets.argmax(1)).sum().item()
                        t_n_total += len(t_outputs)
                        
                test_acc = t_n_correct / t_n_total
                senti_acc /= cnt
                if test_acc > max_test_acc:
                    max_test_acc = test_acc
                print('average_train_acc: {}, average_train_loss: {}, test_acc: {}, senti_acc: {}'.format(avg_train_acc, avg_train_loss, test_acc, senti_acc))
    
    embedding_layer = None
    if params.network_type == 'fasttext':
        embedding_layer = model.embed
    else:
        embedding_layer = model.complex_embed
    
    torch.save(embedding_layer.state_dict(),open('temp/{}_{}'.format(params.network_type, params.dataset_name),'wb'))
    print('max_test_acc: {}'.format(max_test_acc))



if __name__=="__main__":
  
    params = Params()
    config_file = 'config/config_qdnn.ini'    # define dataset in the config
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

