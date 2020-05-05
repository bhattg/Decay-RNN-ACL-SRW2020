import json
import multiprocessing
import os
import sys
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


class EIRNN_Model(object):

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
            key = deps_test[i]['n_diff_intervening']
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

    def log_alpha(self,message):
        with open('logs/alpha_' + self.output_filename, 'a') as file:
            file.write(message + '\n')

    def log_result(self, message):
        with open('logs/result_' + self.output_filename, 'a') as file:
            file.write(message + '\n')        
    
##########################################################
#### EXTERNAL ADDITION DONE TO GET LINZEN TESTED #########
##########################################################

    def external_result_logger(self, result_dict):
        for keys in result_dict.keys():
            message = str(keys)
            message += "Accuracy on {} example is {}".format(result_dict[keys][-1], result_dict[keys][-2])
            self.log_result(message)

    def test_external(self, filename):
    #     #filename contains the pickel files which contains the input and the corresponding target value for different 
    #     #syntactic evaluation. 
        test_data_dict = {}
        for files in os.listdir(filename):
    #         #tuple of list 
            test_pickel = open(os.path.join(filename, files), 'rb')
            test_data_dict[files] = pickle.load(test_pickel)

        results = self.external_testing(test_data_dict)
        self.external_result_logger(results)

    def external_testing(self,d=None):
        testing_result={}
        for files in d.keys():
            X_testing_perFile, Y_testing_perFile = d[files]   #x is list of numpy array
            len_X_testing = len(X_testing_perFile)
            assert len(X_testing_perFile) == len(Y_testing_perFile), "Assert failed at external testing!!"
            predicted=[]
            with torch.no_grad():
                for i in range(len_X_testing):
                    x_test =  X_testing_perFile[i]
                    x_test = torch.tensor(x_test, dtype=torch.long)
                    pred, hidden, output = self.model(x_test)
                    if pred[0][0]> pred[0][1]:
                        predicted.append(0)
                    else:
                        predicted.append(1)
            testing_result[files] = (Y_testing_perFile, predicted, np.sum(np.asarray(Y_testing_perFile)==np.asarray(predicted))/len_X_testing, len_X_testing)
            acc = np.sum(np.asarray(Y_testing_perFile)==np.asarray(predicted))/len_X_testing
            print(str(acc)+" "+str(len_X_testing))

        return testing_result
    
    def load_external_testing(self, filename, save_processed_data = True):
        ex_list = []
        for files in os.listdir(filename):
            pickel = pickle.load(open(os.path.join(filename, files), 'rb'))
            ex_list.append((pickel, files))
        test_example={}
        for i in range(len(ex_list)):
            for keys in ex_list[i][0].keys():
                list1= ex_list[i][0][keys]
                if len(list1[0]) > 2:
                    continue
                if (ex_list[i][1], keys) in test_example.keys():
                    pass
                else:
                    test_example[(ex_list[i][1], keys)]=[]
                for X in list1:            
                    x, x_neg = X
                    test_example[(ex_list[i][1], keys)].append((x, 0))
                    test_example[(ex_list[i][1], keys)].append((x_neg, 1))
        external_testing_dict={}
        for keys in test_example.keys():
            x_test_, y_test_ = zip(*test_example[keys])
            external_testing_dict[keys] = (x_test_, y_test_)
        # At this time we have a dictionary that has key -->(filename, property) and value a tuple  (X_test(string form), y_test)

        final_dict_testing = self.valid_input(external_testing_dict)

        if save_processed_data:
            for keys in final_dict_testing.keys():
                pickle_out = open(os.path.join("Testing_data", str(keys))+".pkl", "wb")
                pickle.dump(final_dict_testing[keys], pickle_out)

        results = self.external_testing(final_dict_testing)
        self.external_result_logger(results)


    def valid_input(self,  external_testing_dict):
        final_dict_testing={}
        for keys in external_testing_dict.keys():
            x = []
            y = []
            X_test, Y_test = external_testing_dict[keys]
            for i in range(len(X_test)):
                x_ex = []
                flag=True
                example = X_test[i]
                token_list = example.split()
                if len(token_list)>self.maxlen:
                    continue
                for tokens in token_list:
                    if not tokens in self.vocab_to_ints.keys():   #if unknown character, leave the example 
                        flag=False
                        break
                    x_ex.append(self.vocab_to_ints[tokens])
                if not flag:
                    continue
                x.append(x_ex)
                y.append(Y_test[i])

            final_dict_testing[keys]=(x, y)
        return final_dict_testing

##########################################################
####  AUTHOR @GANTAVYA BHATT #############################
##########################################################

# IF YOU HAVE AN EXTERNAL LNIZEN .PKL, THEN USE load_data=False, test_external=True, load_external=False
# external_file = address of saved modified pkl inputs 
#pickel folder: address of linzen pikel 

    def pipeline(self, train = True, batched=False, batch_size = 32, shuffle = True, num_workers= 0,
                 load = False, model = '', test_size=7000, 
                 train_size=None, model_prefix='__', epochs=20, data_name='Not', 
                 activation=False, df_name='_verbose_.pkl', load_data=False, 
                 save_data=False, test_external=False, load_external=False, external_file=None, pickel_folder=None):
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
            if test_external:
                if load_external:
                    self.test_external(external_file)
                else :
                    self.load_external_testing(pickel_folder, True)
            else:
                result_dict= self.test_model()
            # print(result_dict)
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

    def no_grad_diag(self):
        for i in range(self.hidden_dim):
            self.model.get_cell(0).weight_hh.grad[i, i]=0
            
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
            self.log_alpha('epoch : ' + str(epoch))
            for index in range(fffstart, len(x_train)) :
                # self.log(index)
                if ((index+1) % 1000 == 0) :
                    self.log(index+1)
                    if ((index+1) % 3000 == 0):
                        acc = self.results()
                        # result_dict = self.result_demarcated()
                        if (acc >= max_acc) :
                            model_name = model_prefix + '.pkl'
                            torch.save(self.model, model_name)
                            max_acc = acc
                    # _ =  self.test_model()
                
                self.model.zero_grad()
                output, hidden, out = self.model(x_train[index])
                if (y_train[index] == 0) :
                    actual = torch.autograd.Variable(torch.tensor([0]), requires_grad=False)#.to(device)
                else :
                    actual = torch.autograd.Variable(torch.tensor([1]), requires_grad=False)#.to(device)
                
                loss = loss_function(output, actual)
                loss.backward(retain_graph=True)

                self.no_grad_diag()
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

