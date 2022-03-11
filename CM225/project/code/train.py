import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import anndata as ad
from sklearn.model_selection import train_test_split
import pickle
from dataset import ModalityMatchingDataset
from model import *
import numpy as np
from scipy.stats import pearsonr
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from torchsummary import summary

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train_and_valid(model,writer, optimizer,scheduler, loss_fn, dataloader_train, dataloader_test, name_model, device):
    best_score = 100000
    for i in range(100):
        train_losses = []
        test_losses = []
        model.train()
        for x, y in dataloader_train:
            optimizer.zero_grad()
            output = model(x.float().to(device))
            loss = loss_fn(output, y.float().to(device))
            #for i in range(3):
            #    loss+= pow(2,-i)*loss_fn(output[i], y.float().to(device))
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()
        scheduler.step()
        test_rmse = []
        test_corr = []
        model.eval()
        with torch.no_grad():
            for x, y in dataloader_test:
                output = model(x.float().to(device))
                #output[output<0] = 0.0
                target = y.float().to(device)
                #print(output.shape,target.shape)
                test_losses.append(loss_fn(output,target).item())
                output = output.cpu().numpy()
                target = target.cpu().numpy()
                test_corr.append(pearsonr(output[0],target[0])[0])
                test_rmse.append(rmse(output[0],target[0]))
            
        if best_score > np.average(test_rmse):
            torch.save(model.state_dict(), name_model+str(i+1)+'.pt')
            best_score = np.average(test_rmse)
        print("Epoch: ",i+1, " Training Loss: ",np.sum(train_losses),"Test Loss: ",np.sum(test_losses), "test rmse: ", np.average(test_rmse),"test corr: ", np.average(test_corr),"best rmse: ", best_score)
        writer.add_scalar('Accuracy/rmse',np.average(test_rmse),i+1)
        writer.add_scalar('Accuracy/corr',np.average(test_corr),i+1)
        writer.add_scalar('Loss/test',np.sum(test_losses),i+1)
        writer.add_scalar('Loss/train',np.sum(train_losses),i+1)
        
    print("best rmse: ", best_score)
    
def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))

def main(args):
    #check gpu available
    if (torch.cuda.is_available()):
        device = 'cuda:0' #switch to current device
        print('current device: gpu')
    else:
        device = 'cpu'
        print('current device: cpu')
    dataset_path = "./data/adt2gex/openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_"
    pretrain_path = args.checkpoint_folder
    
    par = {
        'input_train_mod1': f'{dataset_path}train_mod1.h5ad',
        'input_train_mod2': f'{dataset_path}train_mod2.h5ad',
        'input_test_mod1': f'{dataset_path}test_mod1.h5ad',
        'input_test_mod2': f'{dataset_path}test_mod2.h5ad',
        'output_pretrain': pretrain_path
    }
    
    os.makedirs(par['output_pretrain'], exist_ok=True)
    writer = SummaryWriter(log_dir=args.checkpoint_folder,filename_suffix=args.checkpoint_name)
    print("Start train")
    
    input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
    input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
    train_mod1 = input_train_mod1.to_df()
    train_mod2 = input_train_mod2.to_df()
    test_mod1 = ad.read_h5ad(par['input_test_mod1']).to_df()
    test_mod2 = ad.read_h5ad(par['input_test_mod2']).to_df()
    
    mod1 = input_train_mod1.var['feature_types'][0]
    mod2 = input_train_mod2.var['feature_types'][0]
    
    dataset_train = ModalityMatchingDataset(train_mod1, train_mod2)
    dataloader_train = DataLoader(dataset_train, 64, shuffle = True, num_workers = 0)
    
    dataset_test = ModalityMatchingDataset(test_mod1, test_mod2)
    dataloader_test = DataLoader(dataset_test, 1, shuffle = False, num_workers = 0)
    print("Train Data length: ", len(dataset_train))
    print("Test Data length: ", len(dataset_test))
    
    #model = ModelRegressionAdt2Gex(134,13953).to(device)
    #model = ConvNet4Adt2Gex(134,13953).to(device)
    model = ConvNet10Adt2Gex(134,13953).to(device)
    print(model)
    #summary(model,(1,134))
    
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    loss_fn = torch.nn.MSELoss()
    train_and_valid(model,writer, optimizer,scheduler, loss_fn, dataloader_train, dataloader_test, par['output_pretrain'] + args.checkpoint_name, device)
    print("End train")

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_folder',type=str, default="./pretrain/")
    parser.add_argument('--checkpoint_name',type=str, default="temp")

    args = parser.parse_args()
    main(args)