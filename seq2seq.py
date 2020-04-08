# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:12:54 2020

@author: Stuart
"""
from __future__ import unicode_literals, print_function, division
from io import open
#import unicodedata
#import string
#import re
import random
import time
import numpy as np 
import os
import math 

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
from torch.nn.modules.normalization import LayerNorm

    
class Transformer(nn.Module):
    def __init__(self, input_lex, output_lex, batch_size,
                 d_model=512, nhead=8, num_decoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1, activation="relu" ):

        super(Transformer, self).__init__()
        self.input_lex = input_lex
        self.output_lex = output_lex
        self.batch_size = batch_size
        self.encoder = AttnEncoder(self.input_lex.n_words, d_model, batch_size= self.batch_size)
        self.decoder = AttnDecoder(self.output_lex.n_words, d_model, batch_size= self.batch_size)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead 
        #!!!
        self.linear = nn.Linear(self.d_model, self.output_lex.n_words, bias=False)
        self.softmax = nn.LogSoftmax(dim=2) #nn.Softmax(dim=2)
        

    def forward(self, src_tensor, tgt_tensor, 
                src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """Transforming masked source to target sequences.

        Args:
            src_tensor: the sequence to the encoder (required).
            tgt_tensor: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, N, E)`.
            - tgt: :math:`(T, N, E)`.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.

        Output: 
            - output_gen: generated output.
            - tgt_tensor: embedding target.

        """
        
        #tgt_tensors = self.embedding(tgt.reshape([tgt_seq_len, tgt_batch_size ]).long())
        #output_gen, tgt_tensor = self.decoder(tgt=tgt_tensors, memory=memories, tgt_mask=tgt_mask, memory_mask=memory_mask,
        #                      tgt_key_padding_mask=tgt_key_padding_mask,
        #                      memory_key_padding_mask=memory_key_padding_mask)
        memory = self.encoder(src_tensor)
        gen_tensor = self.decoder(tgt_tensor, memory)
        #gen_tensor = gen_tensor.view(-1, self.d_model)
        print("0", gen_tensor.size())
        output_gen = self.linear(gen_tensor)
        print("1",tgt_tensor.size())
        #output_gen = self.softmax(output_gen).view(-1, self.output_lex.n_words, 1)
        output_gen = self.softmax(output_gen).view(-1, self.output_lex.n_words, self.batch_size)
        print(" output_gen : ", output_gen.size(), " \n")
        
        return output_gen,tgt_tensor #, output_gen.reshape([tgt_batch_size, -1, tgt_seq_len]) 

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. 
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model.
        """
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p) 

    def load_model(self, load_model_name ):
        model_path = "./yourModelPath/%(model_name)s/%(model_name)s.model"%{"model_name":load_model_name}
        self.load_state_dict(torch.load(model_path))
        self.eval()
        print("\n ...model %(model_name)s loaded.\n"%{"model_name":load_model_name} ) 


class AttnEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(AttnEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, input):
        #embedded = self.embedding(input).view(-1, 1, self.hidden_size)
        embedded = self.embedding(input).view(self.batch_size, -1, self.hidden_size)
        print(embedded.size())
        memory = self.transformer_encoder(embedded)
        return memory

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, batch_size):
         super(AttnDecoder, self).__init__()
         self.hidden_size = hidden_size
         self.embedding = nn.Embedding(output_size, self.hidden_size)
         decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=8)
         decoder_norm = LayerNorm(self.hidden_size)
         self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6, norm=decoder_norm)
         
    def forward(self, tgt, memory):
        tgt_tensor = self.embedding(tgt).view(-1, 1, self.hidden_size)
        gen_tensor = self.transformer_decoder(tgt_tensor, memory)
        return gen_tensor#, tgt_tensor

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
    
        