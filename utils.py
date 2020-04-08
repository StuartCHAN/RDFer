# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:51:20 2020

@author: Stuart
"""
import torch
import torch.nn.utils.rnn as rnn_utils
import os
import json
import string
import platform
import pickle


SOS_token = 0
EOS_token = 1


def preprocessData(training_fp, sav_fp, DEVICE, preprocess, reverse=False):
    if preprocess:
        print("\n Initiating dataset from raw files... \n")
        src_dataset, tgt_dataset = loadData(training_fp)
        srclex = Lexicon("src")
        tgtlex = Lexicon("tgt")
        for src in src_dataset:
            srclex.addSentence(src)
        for tgt in tgt_dataset:
            tgtlex.addSentence(tgt)
            
        srclex.saveData()
        tgtlex.saveData()
        
        src_tensors = []
        tgt_tensors = []
        for src in src_dataset:
            indexs = [srclex.word2index[token] for token in src]
            tensor = torch.tensor(indexs, dtype=torch.long, device=DEVICE).view(-1, 1)
            src_tensors.append(tensor)
        for tgt in tgt_dataset:
            indexs = [tgtlex.word2index[token] for token in tgt]
            tensor = torch.tensor(indexs, dtype=torch.long, device=DEVICE).view(-1, 1)
            tgt_tensors.append(tensor)
        
        print("\n SRC number of tokens: ",srclex.n_words, " TGT number of tokens: ", tgtlex.n_words, " \n")
        
        pickle.dump(srclex, open("./dataset/srclex.pkl", "bw"))
        pickle.dump(tgtlex, open("./dataset/tgtlex.pkl", "bw"))

        input_tensors  = rnn_utils.pad_sequence(src_tensors, batch_first=True, padding_value=0)
        target_tensors  = rnn_utils.pad_sequence(tgt_tensors, batch_first=True, padding_value=0)
        
        torch.save(input_tensors, "./dataset/input_tensors.pt")
        torch.save(target_tensors, "./dataset/target_tensors.pt")
    else:
        print("\n Loding dataset from preprocesed files... \n")
        srclex = pickle.load(open("./dataset/srclex.pkl", "br"))
        tgtlex = pickle.load(open("./dataset/tgtlex.pkl", "br"))

        input_tensors  = torch.load("./dataset/input_tensors.pt", map_location=torch.device(DEVICE))
        target_tensors  = torch.load("./dataset/target_tensors.pt", map_location=torch.device(DEVICE))
    
    return srclex, tgtlex, input_tensors, target_tensors;


class Lexicon:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.sents_lens = []
        
    def addSentence(self, tokens):
        self.sents_lens.append(len(tokens))
        for word in tokens:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1 

    def saveData(self):
        json.dump(self.word2index , 
                    open("./dataset/%(name)s.%(n_words)s.word2index.json"%{"name":self.name, "n_words":self.n_words}, "w", encoding="UTF-8") )
        json.dump(self.index2word , 
                    open("./dataset/%(name)s.%(n_words)s.index2word.json"%{"name":self.name, "n_words":self.n_words}, "w", encoding="UTF-8") ) ;

    def max_len(self):
        if len(self.sents_lens) > 0:
            return max(self.sents_lens)

 
"""#These two functions can be active in tasks for Latin-Alphabetic languages.

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s    
"""


def loadData(fp):
    #e.g. fp = "./dataset/train_sample.json"
    data = json.load(open( fp, "r", encoding="UTF-8"))
    src_dataset = []
    tgt_dataset = []
    for item in data:
        inputs = item["input_seq"]
        outputs = item["output_seq"]
        src = []
        for rel in inputs:
            dependentToken = rel["dependentToken"]
            relationship = rel["relationship"]
            targetToken = rel["targetToken"]
            src.extend([dependentToken, relationship, targetToken, "SOS"])
        src.append("EOS")
        tgt = relacement(outputs[:-1])
        src_dataset.append(src)
        tgt_dataset.append(tgt)
    return src_dataset, tgt_dataset


def relacement(tokens):
    tokens = ["SOS" if (word in string.punctuation) else str(word) for word in tokens]
    tokens.append("EOS")
    return tokens


"torch.utils.data.IterableDataset"
class TxtDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensors, target_tensors ):
        self.input_tensors = input_tensors
        self.target_tensors = target_tensors
        assert(len(input_tensors) == len(target_tensors) )
        self.length = len(input_tensors)
 
    def __getitem__(self, index):
        input_tensor = torch.LongTensor(self.input_tensors[index].long()) #torch.FloatTensor(self.input_tensors[index]) #!!! 
        target_tensor = torch.LongTensor(self.target_tensors[index].long()) #!!! 
        #pair = torch.LongTensor(self.train_pairs[index])
        return input_tensor, target_tensor ;  
 
    def __len__(self):
        return self.length ;


def getOSystPlateform():
    sysstr = platform.system()
    if(sysstr =="Windows"):
        return False
    elif(sysstr == "Linux"):
        return True 
    else:
        return False ; 


MODELPATH = "./dataset/model"

def prepare_dir( model_name, stamp):    
    files= os.listdir(MODELPATH)
    models_pool = []
    for file in files: #iterate to get the folders
         if os.path.isdir(MODELPATH+"/"+file): # whether a folder 
              models_pool.append(file)
    savepath = MODELPATH+"/"+model_name+"/"+stamp
    if (model_name not in models_pool) or not( os.path.exists(savepath)) :
        try:
            os.makedirs(savepath)
        except:
            os.makedir(savepath)
    return savepath