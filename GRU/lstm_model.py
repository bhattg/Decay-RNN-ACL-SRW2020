import json
import sys
import multiprocessing
import os
import os.path as op
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import filenames
from utils import deps_from_tsv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
device = cpu

class BatchedDataset(Dataset):
    '''
    This class make a general dataset that we will use to generate 
    the batched training data
    '''
    def __init__(self, x_train, y_train):
        super(BatchedDataset, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        assert (x_train).shape[0] == (y_train).shape[0] 
        self.length =  (x_train).shape[0]
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return self.length


class LSTMModel(object):

    def input_to_string(self, x_input):
        #x_input is the example we want to convert to the string 
        #x_input should be in the form of 1D list. 
        example_string = ""
        for token in x_input:
            if token == 0:
                continue
            str_tok =  self.ints_to_vocab[token]
            example_string+=str_tok+" "
        return example_string



    def demark_testing(self):
        X_test=self.X_test
        Y_test=self.Y_test
        deps_test=self.deps_test
        testing_dict={}
        assert len(X_test)==len(Y_test) and len(Y_test)==len(deps_test)
        for i in (range(len(X_test))):
            key = deps_test[i]['n_intervening']
            if not key in testing_dict.keys():
                testing_dict[key]=[]
            testing_dict[key].append((X_test[i], Y_test[i]))

        self.testing_dict=testing_dict

    serialized_attributes = ['vocab_to_ints', 'ints_to_vocab', 'filename',
                             'X_train', 'Y_train', 'deps_train',
                             'X_test', 'Y_test', 'deps_test']

    def __init__(self, filename=None, serialization_dir=None,
                 batch_size=1, embedding_size=50, hidden_dim = 50,
                 maxlen=50, prop_train=0.9, rnn_output_size=10,
                 mode='infreq_pos', vocab_file=filenames.vocab_file,
                 equalize_classes=False, criterion=None, len_after_verb=0,
                 verbose=1, output_filename='default.txt'):
        '''
        filename: TSV file with positive examples, or None if unserializing
        criterion: dependencies that don't meet this criterion are excluded
            (set to None to keep all dependencies)
        verbose: passed to Keras (0 = no, 1 = progress bar, 2 = line per epoch)
        '''
        self.filename = filename
        self.vocab_file = vocab_file
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.prop_train = prop_train
        self.mode = mode
        self.rnn_output_size = rnn_output_size
        self.maxlen = maxlen
        self.equalize_classes = equalize_classes
        self.criterion = (lambda x: True) if criterion is None else criterion
        self.len_after_verb = len_after_verb
        self.verbose = verbose
        self.output_filename = output_filename
        # self.set_serialization_dir(serialization_dir)

    def log(self, message):
        with open('logs/' + self.output_filename, 'a') as file:
            file.write(str(message) + '\n')

    def log_grad(self, message):
        with open('logs/grad_' + self.output_filename, 'a') as file:
            file.write(message + '\n')

    def log_input(self, message):
        with open('logs/input_' + self.output_filename, 'a') as file:
            file.write(message + '\n')

    def log_forget(self, message):
        with open('logs/forget_' + self.output_filename, 'a') as file:
            file.write(message + '\n')

    def log_output(self, message):
        with open('logs/output_' + self.output_filename, 'a') as file:
            file.write(message + '\n')

    def pipeline(self, train = True, batched=False, batch_size = 32, shuffle = True, num_workers= 0,
                 load = False, model = '', test_size=7000, 
                 train_size=None, model_prefix='__', epochs=20, data_name='Not', 
                 activation=False, df_name='_verbose_.pkl', load_data=False, 
                 save_data=False):
        self.batched= batched
        if (load_data):
            self.load_train_and_test(test_size, data_name)
        else :
            self.log('creating data')
            examples = self.load_examples(data_name, save_data, None if train_size is None else train_size*10)
            self.create_train_and_test(examples, test_size, data_name, save_data)
        if batched:
            self.create_model_batched(batch_size=batch_size)
        else:   
            self.create_model()
        if (load) :
            self.load_model(model)
        if (train) :
            if(batched):
                self.train_batched(epochs, model_prefix, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            else:   
                self.train(epochs, model_prefix)

        else:
            result_dict= self.test_model()
        #    self.cross_validate(0)
            print(result_dict)
        
        print('Data : ',  data_name)
        self.log(data_name)

        if (activation) :
            acc = self.results_verbose(df_name)
        else :
            if self.batched:
                acc= self.results_batched()
            else:
                acc = self.results() 

        if (test_size == -2):
            acctrain = self.results_train()

    def load_examples(self,data_name='Not',save_data=False, n_examples=None):
        '''
        Set n_examples to some positive integer to only load (up to) that 
        number of examples
        '''
        self.log('Loading examples')
        if self.filename is None:
            raise ValueError('Filename argument to constructor can\'t be None')

        self.vocab_to_ints = {}
        self.ints_to_vocab = {}
        examples = []
        n = 0

        deps = deps_from_tsv(self.filename, limit=n_examples)

        for dep in deps:
            tokens = dep['sentence'].split()
            if len(tokens) > self.maxlen or not self.criterion(dep):
                continue

            tokens = self.process_single_dependency(dep)
            ints = []
            for token in tokens:
                if token not in self.vocab_to_ints:
                    # zero is for pad
                    x = self.vocab_to_ints[token] = len(self.vocab_to_ints) + 1
                    self.ints_to_vocab[x] = token
                ints.append(self.vocab_to_ints[token])

            examples.append((self.class_to_code[dep['label']], ints, dep))
            n += 1
            if n_examples is not None and n >= n_examples:
                break

        if (save_data) :
            with open('plus5_v2i.pkl', 'wb') as f:
                pickle.dump(self.vocab_to_ints, f)
            with open('plus5_i2v.pkl', 'wb') as f:
                pickle.dump(self.ints_to_vocab, f)

        return examples

    def load_model(self, model) :
        self.model = torch.load(model)
        
    def train_batched(self, n_epochs=10, model_prefix="__", batch_size=32, shuffle=True, learning_rate=0.002, num_workers=0):
        self.log('Training Batched')
        if not hasattr(self, 'model'):
            self.create_model()
        
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)
        prev_param = list(self.model.parameters())[0].clone()
        max_acc = 0
        self.log(len(self.X_train))
        '''
        Since our Dataset class needs the array as the input and it is actually better to use array as the inputs, 
        so we will conver the training data to array
        '''
        total_batches = int(len(self.X_train)/batch_size)
        x_train = np.asarray(self.X_train)#.to(device)   
        y_train = np.asarray(self.Y_train)#torch.tensor(self.Y_train, requires_grad=False)#.to(device)
        # self.log('cpu to gpu')
        # acc = self.results()
        print("Total Train epochs : "+str(n_epochs))
        print("Total Train batches : "+str(total_batches))

        new_BatchedDataset =  BatchedDataset(x_train, y_train)
        DataGenerator =  DataLoader(new_BatchedDataset, batch_size= batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
        
        for epoch in range(n_epochs) :
            
            self.log('epoch : ' + str(epoch))
            self.log_grad('epoch : ' + str(epoch))
            batches_processed = 0
            batch_list=[]
            for x_batch, y_batch in DataGenerator :
                batch_list.append((x_batch, y_batch))
                if batches_processed!=0 and batches_processed%10==0:
                    self.log("{}/{} Batches Processed".format(batches_processed, total_batches))
                    self.validate_training(batch_list)
                    batch_list=[]
                    # acc =  self.results_batched()
                    # if (acc >= max_acc) :
                    #     model_name = model_prefix + '.pkl'
                    #     torch.save(self.model, model_name)
                    #     max_acc = acc                    

                self.model.zero_grad()
                output, hidden, out = self.model(x_batch)
                loss = loss_function(output,y_batch)
                loss.backward(retain_graph=True)
                optimizer.step()
                batches_processed+=1

                counter = 0
                self.log_grad('batches processed : ' + str(batches_processed))
                for param in self.model.parameters():
                    if param.grad is not None:
                        # print(counter, param.shape)
                        self.log_grad(str(counter) + ' : ' + str(param.grad.norm().item()))
                        counter += 1

            # acc = self.results_batched()
            # if (acc > max_acc) :
            #     model_name = model_prefix + '.pkl'
            #     torch.save(self.model, model_name)
            #     max_acc = acc
            

            
    def train(self, n_epochs=10, model_prefix='__'):
        self.log('Training')
        if not hasattr(self, 'model'):
            self.create_model()
        
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        prev_param = list(self.model.parameters())[0].clone()
        max_acc = 0
        self.log(len(self.X_train))
        x_train = torch.tensor(self.X_train, dtype=torch.long, requires_grad=False)#.to(device)
        y_train = self.Y_train #torch.tensor(self.Y_train, requires_grad=False)#.to(device)
        self.log('cpu to gpu')
        # acc = self.results()
        print(n_epochs)

        fffstart = 0

        for epoch in range(n_epochs) :
            self.log('epoch : ' + str(epoch))
            self.log_grad('epoch : ' + str(epoch))
            
            for index in range(fffstart, len(x_train)) :
                # self.log(index)
                if ((index+1) % 1000 == 0) :
                    self.log(index+1)
                    if ((index+1) % 3000 == 0):
                        acc = self.results()
                        result_dict = self.result_demarcated()
                        self.log(str(result_dict))
                        if (acc >= max_acc) :
                            model_name = model_prefix + '.pkl'
                            torch.save(self.model, model_name)
                            max_acc = acc
                    _ =  self.test_model()
                
                self.model.zero_grad()
                output, hidden, out = self.model(x_train[index])
                if (y_train[index] == 0) :
                    actual = torch.autograd.Variable(torch.tensor([0]), requires_grad=False)#.to(device)
                else :
                    actual = torch.autograd.Variable(torch.tensor([1]), requires_grad=False)#.to(device)
                
                loss = loss_function(output, actual)
                loss.backward(retain_graph=True)
                optimizer.step()

                if ((index) % 10 == 0) :
                    counter = 0
                    self.log_grad('index : ' + str(index))
                    for param in self.model.parameters():
                        if param.grad is not None:
                            # print(counter, param.shape)
                            self.log_grad(str(counter) + ' : ' + str(param.grad.norm().item()))
                            counter += 1

            fffstart = 0

            acc = self.results()
            if (acc > max_acc) :
                model_name = model_prefix + '.pkl'
                torch.save(self.model, model_name)
                max_acc = acc

            # self.results_train()

