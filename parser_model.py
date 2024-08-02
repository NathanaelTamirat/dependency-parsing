#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Feed-Forward Neural Network for Dependency Parsing

import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ParserModel(nn.Module):
    """ Feedforward neural network with an embedding layer and two hidden layers.
    The ParserModel will predict which transition should be applied to a
    given partial parse configuration.
    """
    def __init__(self, embeddings, n_features=36,hidden_size=200, n_classes=3, dropout_prob=0.5):
        super(ParserModel, self).__init__()

        self.n_features = n_features #number of input features
        self.n_classes = n_classes # number of output classes
        self.dropout_prob = dropout_prob #  dropout probability
        self.embeddings = nn.Parameter(torch.tensor(embeddings)) # word embeddings (num_words, embedding_size)
        self.embed_size = embeddings.shape[1] 
        self.hidden_size = hidden_size # number of hidden units
       
        self.embed_to_hidden_weight=nn.Parameter(torch.Tensor(self.n_features*self.embed_size,self.hidden_size))
        self.embed_to_hidden_bias=nn.Parameter(torch.Tensor(self.hidden_size))
        self.hidden_to_logits_weight=nn.Parameter(torch.Tensor(self.hidden_size,self.n_classes))
        self.hidden_to_logits_bias=nn.Parameter(torch.Tensor(self.n_classes))

        nn.init.xavier_uniform_(self.embed_to_hidden_weight)
        nn.init.uniform_(self.embed_to_hidden_bias)
        nn.init.xavier_uniform_(self.hidden_to_logits_weight)
        nn.init.uniform_(self.hidden_to_logits_bias)

        self.dropout=nn.Dropout(p=self.dropout_prob)

    def embedding_lookup(self, w):
        """ Utilize `w` to select embeddings from embedding matrix `self.embeddings`
            @param w (Tensor): input tensor of word indices (batch_size, n_features)

            @return x (Tensor): tensor of embeddings for words represented in w
                                (batch_size, n_features * embed_size)
        """
        x=self.embeddings[w]
        x=x.view(w.shape[0],-1) # (batch_size, n_features * embed_size)
        return x
    
    def forward(self, w):
        """
        @param w (Tensor): input tensor of tokens (batch_size, n_features)

        @return logits (Tensor): tensor of predictions (output after applying the layers of the network)
                                 without applying softmax (batch_size, n_classes)
        """
        embed=self.embedding_lookup(w)
        hidden=F.relu(torch.matmul(embed,self.embed_to_hidden_weight) + self.embed_to_hidden_bias)
        hid_dropout=self.dropout(hidden)
        logits=torch.matmul(hid_dropout,self.hidden_to_logits_weight)+self.hidden_to_logits_bias
        return logits

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Simple sanity check for parser_model.py')
    parser.add_argument('-e', '--embedding', action='store_true', help='sanity check for embeding_lookup function')
    parser.add_argument('-f', '--forward', action='store_true', help='sanity check for forward function')
    args = parser.parse_args()

    embeddings = np.zeros((100, 30), dtype=np.float32)
    model = ParserModel(embeddings)

    def check_embedding():
        inds = torch.randint(0, 100, (4, 36), dtype=torch.long)
        selected = model.embedding_lookup(inds)
        assert np.all(selected.data.numpy() == 0), "The result of embedding lookup: " \
                                      + repr(selected) + " contains non-zero elements."

    def check_forward():
        inputs =torch.randint(0, 100, (4, 36), dtype=torch.long)
        out = model(inputs)
        expected_out_shape = (4, 3)
        assert out.shape == expected_out_shape, "The result shape of forward is: " + repr(out.shape) + \
                                                " which doesn't match expected " + repr(expected_out_shape)

    if args.embedding:
        check_embedding()
        print("Embedding_lookup sanity check passes!")

    if args.forward:
        check_forward()
        print("Forward sanity check passes!")
