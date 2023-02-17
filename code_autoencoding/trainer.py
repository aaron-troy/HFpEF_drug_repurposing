import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

def train_AE(AE, train_loader, val_loader, method = 'Adam', n_epochs = 1000, learning_rate = 1e-4, max_patience = 20, stop_threshold = 1e-15, train_id = ''):

    # The AE can be trained using either Adam or stochasitc gradient descent. Here we use Adam
    if method == 'Adam':
        optimizer = torch.optim.Adam(AE.parameters(), lr=learning_rate)
    elif method == 'SGD':
        optimizer = torch.optim.SGD(AE.parameters(), lr=learning_rate)
    else:
        print('Unknown optimization method. Choose Adam or SGD')
        return False

    AE.cuda()

    patience_cnt = 0
    loss = np.float('inf')

    for e in range(n_epochs):

        print('Epoch: ', e)

        train_loss = train_step(AE, optimizer, train_loader)
        print('Train loss:', train_loss)

        test_loss = val_step(AE, val_loader)
        print('Test loss:', test_loss)

        if test_loss < loss:
            loss = test_loss
            patience_cnt = 0
            #AE.cpu()
            # TODO: Add functionality for saving updated model parameters, move back to CUDA afterwards

            #AE.cuda()
        else:
            patience_cnt += 1


        if (loss < stop_threshold) or (patience_cnt > max_patience):
            print('Best loss: ', loss)
            break

    AE.cpu()
    d = dict()
    d['state_dict'] = AE.state_dict()
    torch.save(d,  str(round(loss, 8)) + '_' + train_id + '_model_best.pth')
    AE.cuda()
    return False

def train_step(AE, optimizer, train_loader):

    # Put in train mode
    AE.train()
    criterion = nn.MSELoss()
    train_loss = 0

    for idx, batch in enumerate(train_loader):
        #print("Batch: ", idx)
        optimizer.zero_grad()

        inputs = Variable(batch[1]).cuda()
        output = AE(inputs)

        loss = criterion(output, inputs)
        loss.backward()

        optimizer.step()
        train_loss += loss.cpu().data.numpy() * len(inputs)

    return train_loss / len(train_loader.dataset)

def val_step(AE, val_loader, prt = False):

    #Put in eval mode
    AE.eval()

    #Set loss criterion and initial loss
    criterion = torch.nn.MSELoss()
    val_loss = 0

    for idx, batch in enumerate(val_loader):
        inputs = Variable(batch[1]).cuda()
        with torch.no_grad():
            output = AE(inputs)
        loss = criterion(output, inputs)

        val_loss += loss.item() * len(inputs)

        if(prt):
            print(loss.item() * len(inputs))

    return val_loss / len(val_loader.dataset)