# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:01:36 2020

@author: Stuart
"""
import torch 
import torch.nn as nn 
import argparse
import utils
import seq2seq
import optimization

r"Device check as default setting."
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("* -- Begining model with %s -- *\n"%DEVICE)

if __name__ == "__main__":
 
    parser = argparse.ArgumentParser(description=" RDFer Model00 -- by Stuart ")
    parser.add_argument('--train_dataset', type=str, default="./dataset/train_sample.json")
    parser.add_argument('--save_dataset', type=str, default="./dataset/train_sample.pt")
    parser.add_argument('--train', type=bool, default = False)
    parser.add_argument('--preprocess', type=bool, default = False)
    parser.add_argument('--model_name', type=str, default = "SmoothNLP")
    parser.add_argument('--batch_size', type=int, default = 62 )
    parser.add_argument('--num_iter', type=int, default = 100 )
    parser.add_argument('--learning_rate', type=float, default = 0.01 )
    args = parser.parse_args()
    
    training_fp = args.train_dataset
    sav_fp = args.save_dataset 
    train = args.train
    preprocess = args.preprocess
    model_name = args.model_name if args.model_name is not None else str().join(str(training_fp.split("/")[-1]).split(".")[:-1] )
    batch_size = args.batch_size
    n_iter = args.num_iter
    learning_rate = args.learning_rate
    
    if preprocess:
        srclex, tgtlex, input_tensors, target_tensors = utils.preprocessData(training_fp, sav_fp, DEVICE, preprocess=preprocess, reverse=False)
        
    if train:
        model = seq2seq.Transformer(srclex, tgtlex,  batch_size= batch_size)
        optimization.iterTrain(input_tensors, target_tensors, model, n_iter, batch_size, learning_rate, mom=0, model_name=model_name ) 

        

