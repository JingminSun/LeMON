import numpy as np
import torch
from torch import nn, optim
from numpy import linalg as LA
import time
import h5py
import yaml
import logging
import os

# activation2 = nn.ReLU()
activation = nn.Tanh()

class OneInputBasis(nn.Module):
    def __init__(self,num_sensors,dim1):
        super().__init__()
        self.num_sensors = num_sensors
        self.dim1 = dim1

        bo_b = True
        bo_last = False

        self.l1 = nn.Linear(self.num_sensors, 100, bias=bo_b)
        self.l4 = nn.Linear(100,self.dim1, bias=bo_last)

    def forward(self, v):
        v = activation(self.l1(v))
        v = (self.l4(v))

        return v


class node(nn.Module):
    def __init__(self,dim_output_space_basis,num_sensors,dim1):
        super().__init__()
        self.dim_output_space_basis = dim_output_space_basis
        self.num_sensors = num_sensors
        self.dim1 = dim1

        self.set_lay = nn.ModuleList([OneInputBasis(self.num_sensors,self.dim1) for _ in range(self.dim_output_space_basis)])


    def forward(self, v):

        w = self.set_lay[0](v)

        for ii in range(self.dim_output_space_basis - 1):
            w = torch.cat((w, self.set_lay[ii + 1](v)), dim=1)

        return w


class mesh(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        bo_b = True
        bo_last = False

        self.l3 = nn.Linear(2, 100, bias=bo_b)
        self.l4 = nn.Linear(100, 100, bias=bo_b)
        self.l5 = nn.Linear(100, 100, bias=bo_b)
        self.l6 = nn.Linear(100, 100, bias=bo_b)
        # self.l8 = nn.Linear(100, 100, bias=bo_b)
        # self.l9 = nn.Linear(100, 100, bias=bo_b)
        self.l7 = nn.Linear(100, self.hidden_dim * self.output_dim, bias=bo_last)

    def forward(self, w):
        w = activation(self.l3(w))
        w = activation(self.l4(w))
        w = activation(self.l5(w))
        w = activation(self.l6(w))
        # w = activation(self.l8(w))
        # w = activation(self.l9(w))
        w = (self.l7(w))

        return w


class DeepONet(nn.Module):
    def __init__(self,model_config, data_config):
        super().__init__()

        self.input_dim = data_config.x_num
        self.num_sensors = data_config.input_len //data_config.input_step
        self.dim_output_space_basis = model_config.basis_dim
        self.output_dim = 1

        # self.toplayer = nn.ModuleList(
        #     [node(self.dim_output_space_basis, self.num_sensors) for _ in range(self.input_dim)])
        self.top = node(self.dim_output_space_basis,self.num_sensors * self.input_dim,1)

        self.bottom = mesh(self.dim_output_space_basis, self.output_dim)

    def forward(self, querypoint, value_at_sensor):
        querypoint = self.bottom(querypoint)
        value_at_sensor = value_at_sensor.view(-1,1,self.num_sensors * self.input_dim)
        k1 = self.top(value_at_sensor)
        k1 = k1.view(-1, self.dim_output_space_basis, 1)
        output = torch.bmm(querypoint, k1)

        # e = torch.bmm(w ,k)
        # e =  e[:, :, 0]
        return output

