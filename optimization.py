# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:56:44 2020

@author: Stuart
"""
import time
import torch
import torch.utils.data as Data
import utils
import plotting


def iterTrain(input_tensors, target_tensors, model, n_iters, batch_size, learning_rate, mom=0, model_name="RDFer"):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=mom)
    criterion = torch.nn.NLLLoss()
    torch_dataset = utils.TxtDataset(input_tensors, target_tensors)
    
    r""" Put the dataset into DataLoader
    """"
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,  # MINIBATCH_SIZE = 6
        shuffle=True,
        drop_last= False,
        num_workers= 2 if utils.getOSystPlateform() else 0  # set multi-work num read data based on OS plateform
        #collate_fn= utils.collate_fn  #!!! 
    ) 
    print(" Dataset loader ready, begin training. \n") 
    
    datset_len = len(loader)
    print("\n Dataset loader length is ", datset_len, ", save model every batch. " )
    losses = []
    for epoch in range(1, n_iters + 1):
        # an epoch goes the whole data
        for batch, (input_tensor, tgt_tensor) in enumerate(loader):
            # here to train your model
            input_tensor, tgt_tensor = input_tensor.view(-1, batch_size).long(), tgt_tensor.view(-1, batch_size).long()
            print('\n\n  - Epoch ', epoch, ' | batch ', batch, '\n | input lenght:   ', input_tensor.size(), '\n | target length:   ', tgt_tensor.size() ," \n")  
            loss = optimize(input_tensor, tgt_tensor, model, optimizer, criterion)
            print(" loss:", loss)
            with open('./dataset/model/%(model_name)s.txt'%{ "model_name":model_name}, "a", encoding="UTF-8") as save:
                save.write(str(loss)+"\n")
                save.close();
        stamp = save_model(model)
        plotting.showPlot(losses, model_name, stamp)
    return model,losses


def optimize(input_tensor, tgt_tensor, model, optimizer, criterion):
    optimizer.zero_grad()
    gen_tensor, tgt_tensor = model(input_tensor, tgt_tensor)
    loss = criterion(gen_tensor, tgt_tensor)
    loss.backward()
    optimizer.step()
    return loss.item()


def save_model(model):
    stamp= str(time.strftime('%Y_%m_%d_%H_%M_%S', time.gmtime(time.time())))
    torch.save(model.state_dict(), "./dataset/model/%s.model"%stamp )
    return stamp

    
    
    
    