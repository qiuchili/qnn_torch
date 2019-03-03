# -*- coding: utf-8 -*-

from params import Params
import dataset
import numpy as np
import torch
import torch.nn as nn
import models
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import pickle

def run(params):
    params.network_type = 'mlp'

    pkl_name="temp/"+params.dataset_name+".alphabet.pkl"
    dictionary = pickle.load(open(pkl_name,'rb'))
    sentiment_dic = build_sentiment_lexicon(dictionary, params.sentiment_dic_file)
    train_index_array, test_index_array = k_fold_split(sentiment_dic, k=params.n_fold)
    
    
    network_file = "temp/"+"{}_{}".format(params.model_type,params.dataset_name)
    weights_dic = torch.load(network_file, map_location=params.device_type)
    embedding = None
    if params.variant == 'phase':
        embedding = weights_dic['phase_embed.weight']
    
    elif params.variant == 'amplitude':
        embedding = weights_dic['amplitude_embed.weight']
    
    elif params.variant == 'complex':
        #Currently unavailable, as complex structures needed to be implemented to process
        phase_embedding = weights_dic['phase_embed.weight']
        amplitude_embedding = weights_dic['amplitude_embed.weight']
        re = amplitude_embedding *torch.cos(phase_embedding)
        im = amplitude_embedding *torch.sin(phase_embedding)
        embedding = re,im
    elif params.variant == 'real':
        embedding = weights_dic['weight']
    else:
        embedding = weights_dic['phase_embed.weight']
    
    
    params.embedding_size = embedding.shape[1]
    params.nb_classes = 1
    #Construct training and test set
    model = models.setup(params)
    model = model.to(params.device)
    model._reset_params()
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(list(model.parameters()), lr=params.lr)
    accuracy_list = []
    fold_id = 0
    for train_index, test_index in zip(train_index_array, test_index_array):
        print('fold {}:'.format(fold_id))
        fold_id = fold_id+1
        train_x = embedding[train_index[:,0]]
        train_y = sentiment_dic[train_index[:,0],0]
        train_y = torch.tensor(train_y, dtype = torch.float32).unsqueeze(1)
        test_x = embedding[test_index[:,0]]
        test_y = sentiment_dic[test_index[:,0],0]
        test_y = torch.tensor(test_y, dtype = torch.float32).unsqueeze(1)
        trainDataset = TensorDataset(train_x,train_y)
        testDataset = TensorDataset(test_x,test_y)
        train_loader = DataLoader(trainDataset, batch_size = params.batch_size, shuffle = True)
        test_loader = DataLoader(testDataset, batch_size = params.batch_size, shuffle = False)
#        testDataset = TensorDataset(test_x,test_y)
        for i in range(params.epochs):
            print('epoch: ', i)
            for _i,data in enumerate(train_loader,0):
                train_accs = []
                train_losses = []
                batch_x, batch_y = data
                inputs, targets = batch_x.to(params.device),batch_y.to(params.device)
                model.train()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                n_correct = (outputs.sign() == targets).sum().item()
                n_total = len(outputs)
                train_acc = n_correct / n_total
                train_accs.append(train_acc)
                train_losses.append(loss.item())    
#                print(train_acc,loss.item())
#                if _i % 5 == 0:
#                    model.eval()
#                    avg_train_acc = np.mean(train_accs)
#                    avg_train_loss = np.mean(train_losses)
#                    train_accs = []
#                    train_losses = []
#                    t_n_correct = 0
#                    t_n_total = 0
#                    cnt = 0
#                    for _ii,test_data in enumerate(test_loader,0):
#                        cnt += 1
#                        test_x, test_y = test_data
#                        test_output = model(test_x)
#                        t_n_correct += (test_output.sign() == test_y).sum().item()
#                        t_n_total += len(test_output)
#                    test_acc = t_n_correct / t_n_total
#                    print('average_train_acc: {}, average_train_loss: {}, test_acc: {}'.format(avg_train_acc, avg_train_loss, test_acc))
        model.eval()
        t_n_correct = 0
        t_n_total = 0
        cnt = 0
        for _ii,test_data in enumerate(test_loader,0):
            cnt += 1
            test_x, test_y = test_data
            test_output = model(test_x)
            t_n_correct += (test_output.sign() == test_y).sum().item()
            t_n_total += len(test_output)
        test_acc = t_n_correct / t_n_total
        accuracy_list.append(test_acc)
        print('final test accuracy: {}'.format(test_acc))
        model._reset_params()
    print('cross_validation accuracy: {}'.format(sum(accuracy_list)/float(len(accuracy_list))))
        
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
 
    

def build_sentiment_lexicon(dictionary, sentiment_dic_file):
    sentiment_lexicon = np.zeros((len(dictionary),1))
    with open(sentiment_dic_file, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        for line in fin:
            tokens = line.rstrip().split()
            if tokens[0] in dictionary:
                index = dictionary[tokens[0]]
                sentiment_lexicon[index] = np.asarray(tokens[1:], dtype='float32')
    return sentiment_lexicon
    

    
def k_fold_split(sentiment_dic,k=5):
    non_zero_ind = np.argwhere(sentiment_dic[:,0])
    kf = KFold(n_splits=k)
    train_index_array = []
    test_index_array = []
    for train_index, test_index in kf.split(non_zero_ind):
        train_index_array.append(non_zero_ind[train_index])
        test_index_array.append(non_zero_ind[test_index])
    return train_index_array, test_index_array



if __name__=="__main__":
  
    params = Params()
    config_file = 'config/config_cv.ini'    # define dataset in the config
    params.parse_config(config_file)    
    
    if torch.cuda.is_available():
        params.device = torch.device('cuda')
        params.device_type = 'cuda'
        torch.cuda.manual_seed(params.seed)
    else:
        params.device = torch.device('cpu')
        params.device_type = 'cpu'
        torch.manual_seed(params.seed)
    
    run(params)

