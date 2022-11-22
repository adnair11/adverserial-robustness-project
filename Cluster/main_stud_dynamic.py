#Import libraries

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models
import torchvision.utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

#from torchattacks import PGD, FGSM, FFGSM
import rep_transformations as rt
from no_attack import NO_ATTACK
from stochastic_attack import STOCHASTIC_ATTACK
from fgsm_ import FGSM_
from ffgsm_ import FFGSM_
from slide_ import SLIDE_
from pgd_ import PGD_
from pgdl2_ import PGDL2_
from data_transformations import DctBlurry


#Show CUDA settings

print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
  
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device: {cuda_id}")

cuda_name = torch.cuda.get_device_name(cuda_id)
print(f"Name of current CUDA device: {cuda_name}")
print(f"CUDA devide properties: {torch.cuda.get_device_properties(0)}")


#Downloading Fashion MNIST Dataset
train_list_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

test_list_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

fashion_mnist_train = dsets.CIFAR10(root='conda/SRP_cluster/data',
                          train=True,
                          transform=train_list_transforms,
                          download=False)

fashion_mnist_aux = dsets.CIFAR10(root='conda/SRP_cluster/data',
                         train=False,
                         transform=test_list_transforms,
                         download=False)

fashion_mnist_test, fashion_mnist_valid = torch.utils.data.random_split(fashion_mnist_aux, [5000, 5000], generator=torch.Generator().manual_seed(42))


#Initialize DataLoader

batch_size = 128
torch.manual_seed(0)
n_workers = 5

train_loader  = torch.utils.data.DataLoader(dataset=fashion_mnist_train,
                                           batch_size=batch_size,
                                           shuffle=42, num_workers=n_workers)

test_loader = torch.utils.data.DataLoader(dataset=fashion_mnist_test,
                                         batch_size=batch_size,
                                         shuffle=42, num_workers=n_workers)

valid_loader = torch.utils.data.DataLoader(dataset=fashion_mnist_valid,
                                         batch_size=batch_size,
                                         shuffle=42, num_workers=n_workers)

#Initialize transformations

id_transf = rt.Identity()
fft_transf = rt.FFT()
dct_transf = rt.DCT()
jpeg_transf = rt.JPEG(block_size=8)
log_transf = rt.LOG()

list_transf = [id_transf, fft_transf, dct_transf, jpeg_transf,log_transf]


#Get model and optimizer

def generate_model(weights=None, lr=0.001):

    #Init model and optimizer
    model = torchvision.models.resnet50(weights=weights).cuda() #CNN.cuda() to resnet 50
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    #Change model architecture 
    model.fc = nn.Sequential(nn.Linear(2048, 256), 
                              nn.ReLU(), 
                              nn.Linear(256, 10))
    
    model.load_state_dict(torch.load("conda/SRP_cluster/model_weights"))
    
    #Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.to(device)
        
    return model, optimizer


#modified test function with 3 attacks and toggle between Pixel and DCT representation.Default is Pixel

def get_accuracy(model, test_loader, atk):
    
    model.eval()
    correct = 0
    total = 0    

    for images, labels in test_loader:

        images = atk(images, labels).cuda()              
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()

    return 100 * float(correct) / total


#train function with PGD attack and toggle between DCT and Pixel representation .Default is Pixel

def train_(model, train_loader, optimizer, loss, atk, num_epochs=5):

    model.train()

    total_batch = len(fashion_mnist_train) // batch_size

    for epoch in range(num_epochs):
        
        for i, (batch_images, batch_labels) in enumerate(train_loader):

           
            Y = batch_labels.cuda()
            X = atk(batch_images, batch_labels).cuda()

            pre = model(X)
            cost = loss(pre, Y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Train. Atk: {atk.attack}, Epoch [{epoch+1}/{num_epochs}], lter [{i+1}/{total_batch}], Loss: {cost.item()}')
    return cost


def round_robin(model, train_loader, optimizer, loss, atks_list, num_epochs=5):

    model.train() 
    num_attacks = len(atks_list)
    total_batch = len(fashion_mnist_train) // batch_size

    for epoch in range(num_epochs): #+1 to round up

        atk_list = random.sample(range(16),6)
        atks_list = generate_atks(model, atk_list, eps=0.06)
        
        count = 0

        for i, (batch_images, batch_labels) in enumerate(train_loader):

            atk = atks_list[count]
            Y = batch_labels.cuda()
            X = atk(batch_images, batch_labels).cuda()

            pre = model(X)
            cost = loss(pre, Y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            count += 1
            count %= num_attacks

            if (i+1) % 100 == 0:
                print(f'RR. Epoch [{epoch+1}/{num_epochs}], lter [{i+1}/{total_batch}], Loss: {cost.item()}')
    return cost


def greedy(model, train_loader, valid_loader, optimizer, loss, atks_list, num_epochs=5):

    num_attacks = len(atks_list)
    total_batch = len(fashion_mnist_train) // batch_size

    for epoch in range(num_epochs):

        atk_list = random.sample(range(16),6)
        atks_list = generate_atks(model, atk_list, eps=0.06)

        loss_list = [0]*num_attacks

        model.eval()

        for i, (batch_images, batch_labels) in enumerate(valid_loader):

            Y = batch_labels.cuda()
            current_batch_size = Y.shape[0]

            for i, atk in enumerate(atks_list):

                X = atk(batch_images, batch_labels).cuda()

                pre = model(X)
                cost = loss(pre, Y)

                loss_list[i] += float(cost)/current_batch_size

        max_loss_arg = np.argmax(np.array(loss_list))
        atk = atks_list[max_loss_arg]

        print(f"Greedy. Epoch [{epoch+1}/{num_epochs}], Worst loss attack: {atk.attack}, loss: {loss_list[max_loss_arg]}")

        model.train()

        for i, (batch_images, batch_labels) in enumerate(train_loader):

            Y = batch_labels.cuda()
            X = atk(batch_images, batch_labels).cuda()

            pre = model(X)
            cost = loss(pre, Y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Greedy. Epoch [{epoch+1}/{num_epochs}], lter [{i+1}/{total_batch}], Loss: {cost.item()}')
    return cost


def multiplicative_weights(model, train_loader, valid_loader, optimizer, loss, atks_list, num_epochs=5):

    num_attacks = len(atks_list)
    total_batch = len(fashion_mnist_train) // batch_size
    w_list = [1]*num_attacks
    eta = 0.1
    

    for epoch in range(num_epochs):

        atk_list = random.sample(range(16),6)
        atks_list = generate_atks(model, atk_list, eps=0.06)

        loss_list = [0]*num_attacks
        model.eval()

        for i, (batch_images, batch_labels) in enumerate(valid_loader):

            Y = batch_labels.cuda()
            current_batch_size = Y.shape[0]

            for i, atk in enumerate(atks_list):

                X = atk(batch_images, batch_labels).cuda()

                pre = model(X)
                cost = loss(pre, Y)

                loss_list[i] += float(cost)/current_batch_size

        loss_list = np.e**(eta*np.array(loss_list))
        w_list *= loss_list
        total_loss = w_list.sum()
        prob_list = np.cumsum(w_list)
        prob_list /= total_loss

        log = f"MP. Epoch [{epoch+1}/{num_epochs}], atks: "
        for i, atk in enumerate(atks_list):
            log += f"({atk.attack},{prob_list[i]}) "

        print(log)

        model.train()

        for i, (batch_images, batch_labels) in enumerate(train_loader):

            rand = random.random()
            atk_idx = 0
            for j in range(1,num_attacks):
                if rand > prob_list[j]:
                    atk_idx = j
                else:
                    break

            atk = atks_list[atk_idx]
            Y = batch_labels.cuda()
            X = atk(batch_images, batch_labels).cuda()

            pre = model(X)
            cost = loss(pre, Y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'MP. Epoch [{epoch+1}/{num_epochs}], lter [{i+1}/{total_batch}], Loss: {cost.item()}')
    return cost


def get_model_acc(model, test_loader, atks_list):
    
    model_acc = {}
    
    for atk in atks_list:
        model_atk_acc = get_accuracy(model, test_loader, atk)
        model_acc[atk.attack] = model_atk_acc
        
    return model_acc


#Define attacks_list generation

def generate_atks(model_atk, attack_list, eps=0.06, eps2=0.1):
    
    atks_list = []
    atks_list.append(NO_ATTACK(model_atk))
    atks_list.append(STOCHASTIC_ATTACK(model_atk, eps=eps))
    
    if 0 in attack_list:
        atks_list.append(FGSM_(model_atk, eps=eps))
    if 1 in attack_list:
        atks_list.append(FGSM_(model_atk, eps=eps, transf=dct_transf))
    if 2 in attack_list:
        atks_list.append(FGSM_(model_atk, eps=eps, transf=jpeg_transf))
    if 3 in attack_list:
        atks_list.append(FGSM_(model_atk, eps=eps, transf=fft_transf))
        
    if 4 in attack_list:
        atks_list.append(FFGSM_(model_atk, eps=eps))
    if 5 in attack_list:
        atks_list.append(FFGSM_(model_atk, eps=eps, transf=dct_transf))
    if 6 in attack_list:
        atks_list.append(FFGSM_(model_atk, eps=eps, transf=jpeg_transf))
    if 7 in attack_list:
        atks_list.append(FFGSM_(model_atk, eps=eps, transf=fft_transf))
        
    if 8 in attack_list:
        atks_list.append(PGD_(model_atk, eps=eps))
    if 9 in attack_list:
        atks_list.append(PGD_(model_atk, eps=eps, transf=dct_transf))
    if 10 in attack_list:
        atks_list.append(PGD_(model_atk, eps=eps, transf=jpeg_transf))
    if 11 in attack_list:
        atks_list.append(PGD_(model_atk, eps=eps, transf=fft_transf))
        
    if 12 in attack_list:
        atks_list.append(PGDL2_(model_atk, eps=eps2))
    if 13 in attack_list:
        atks_list.append(PGDL2_(model_atk, eps=eps2, transf=dct_transf))
    if 14 in attack_list:
        atks_list.append(PGDL2_(model_atk, eps=eps2, transf=jpeg_transf))
    if 15 in attack_list:
        atks_list.append(PGDL2_(model_atk, eps=eps2, transf=fft_transf))
        
    return atks_list


#Define attacks_list generation

def generate_atks_test(model_atk, eps=0.06, eps2=0.1):
    
    atks_list = []
    atks_list.append(NO_ATTACK(model_atk))
    atks_list.append(STOCHASTIC_ATTACK(model_atk, eps=eps))
    
    atks_list.append(FGSM_(model_atk, eps=eps))
    atks_list.append(FGSM_(model_atk, eps=eps, transf=dct_transf))
    atks_list.append(FGSM_(model_atk, eps=eps, transf=jpeg_transf))
    atks_list.append(FGSM_(model_atk, eps=eps, transf=fft_transf))
    
    atks_list.append(FFGSM_(model_atk, eps=eps))
    atks_list.append(FFGSM_(model_atk, eps=eps, transf=dct_transf))
    atks_list.append(FFGSM_(model_atk, eps=eps, transf=jpeg_transf))
    atks_list.append(FFGSM_(model_atk, eps=eps, transf=fft_transf))
    
    atks_list.append(PGD_(model_atk, eps=eps))
    atks_list.append(PGD_(model_atk, eps=eps, transf=dct_transf))
    atks_list.append(PGD_(model_atk, eps=eps, transf=jpeg_transf))
    atks_list.append(PGD_(model_atk, eps=eps, transf=fft_transf))
    
    atks_list.append(PGDL2_(model_atk, eps=eps2))
    atks_list.append(PGDL2_(model_atk, eps=eps2, transf=dct_transf))
    atks_list.append(PGDL2_(model_atk, eps=eps2, transf=jpeg_transf))
    atks_list.append(PGDL2_(model_atk, eps=eps2, transf=fft_transf))
        
    return atks_list


def get_model_avg(n_models=1, procedure="None", num_epochs=5, eps=0.06, eps2=0.1, r_start=True):

    loss = nn.CrossEntropyLoss()
    iters_result = {}
    atk_list = random.sample(range(16),6)
    print('Atk list idx:', atk_list)
    
    for i in range(n_models):
        model, optim = generate_model()
        atks_model = generate_atks(model, atk_list, eps=eps)
        atks_model_test = generate_atks_test(model, eps=eps)
        
        if iters_result == {}:
            for atk in atks_model_test:
                iters_result[atk.attack] = []

        cost0=train_(model,train_loader,optim,loss,atks_model_test[0], num_epochs=10)

        if procedure == "RR":
            cost1 = round_robin(model,train_loader,optim,loss,atks_model, num_epochs=num_epochs)
        elif procedure == "Greedy":
            cost1 = greedy(model,train_loader,valid_loader,optim,loss,atks_model, num_epochs=num_epochs)
        elif procedure == "MW":
            cost1 = multiplicative_weights(model,train_loader,valid_loader,optim,loss,atks_model, num_epochs=num_epochs)
            
        elif procedure == "FGSM":
            atk = FGSM_(model, eps=eps)
            cost1=train_(model,train_loader,optim,loss,atk, num_epochs=num_epochs)
        elif procedure == "FGSM_DCT":
            atk = FGSM_(model, eps=eps, transf=dct_transf)
            cost1=train_(model,train_loader,optim,loss,atk, num_epochs=num_epochs)
        elif procedure == "FGSM_JPEG":
            atk = FGSM_(model, eps=eps, transf=jpeg_transf)
            cost1=train_(model,train_loader,optim,loss,atk, num_epochs=num_epochs)
        elif procedure == "FGSM_FFT":
            atk = FGSM_(model, eps=eps, transf=fft_transf)
            cost1=train_(model,train_loader,optim,loss,atk, num_epochs=num_epochs)
        elif procedure == "FGSM_LOG":
            atk = FGSM_(model, eps=eps, transf=log_transf)
            cost1=train_(model,train_loader,optim,loss,atk, num_epochs=num_epochs)
            
        elif procedure == "FFGSM":
            atk = FFGSM_(model, eps=eps)
            cost1=train_(model,train_loader,optim,loss,atk, num_epochs=num_epochs)
        elif procedure == "FFGSM_DCT":
            atk = FFGSM_(model, eps=eps, transf=dct_transf)
            cost1=train_(model,train_loader,optim,loss,atk, num_epochs=num_epochs)
        elif procedure == "FFGSM_JPEG":
            atk = FFGSM_(model, eps=eps, transf=jpeg_transf)
            cost1=train_(model,train_loader,optim,loss,atk, num_epochs=num_epochs)
        elif procedure == "FFGSM_FFT":
            atk = FFGSM_(model, eps=eps, transf=fft_transf)
            cost1=train_(model,train_loader,optim,loss,atk, num_epochs=num_epochs)
        elif procedure == "FFGSM_LOG":
            atk = FFGSM_(model, eps=eps, transf=log_transf)
            cost1=train_(model,train_loader,optim,loss,atk, num_epochs=num_epochs)
        
            
        elif procedure == "PGD":
            atk = PGD_(model, eps=eps, steps=20, random_start=r_start)
            cost1=train_(model,train_loader,optim,loss,atk, num_epochs=num_epochs)
        elif procedure == "PGD_DCT":
            atk = PGD_(model, eps=eps, steps=20, transf=dct_transf, random_start=r_start)
            cost1=train_(model,train_loader,optim,loss,atk, num_epochs=num_epochs)
        elif procedure == "PGD_JPEG":
            atk = PGD_(model, eps=eps, steps=20, transf=jpeg_transf, random_start=r_start)
            cost1=train_(model,train_loader,optim,loss,atk, num_epochs=num_epochs)
        elif procedure == "PGD_FFT":
            atk = PGD_(model, eps=eps, steps=20, transf=fft_transf, random_start=r_start)
            cost1=train_(model,train_loader,optim,loss,atk, num_epochs=num_epochs)
        elif procedure == "PGD_LOG":
            atk = PGD_(model, eps=eps, steps=20, transf=log_transf, random_start=r_start)
            cost1=train_(model,train_loader,optim,loss,atk, num_epochs=num_epochs)
            
        elif procedure == "PGDL2":
            atk = PGDL2_(model, eps=eps2, steps=20, random_start=r_start)
            cost1=train_(model,train_loader,optim,loss,atk, num_epochs=num_epochs)
        elif procedure == "PGDL2_DCT":
            atk = PGDL2_(model, eps=eps2, steps=20, transf=dct_transf, random_start=r_start)
            cost1=train_(model,train_loader,optim,loss,atk, num_epochs=num_epochs)
        elif procedure == "PGDL2_JPEG":
            atk = PGDL2_(model, eps=eps2, steps=20, transf=jpeg_transf, random_start=r_start)
            cost1=train_(model,train_loader,optim,loss,atk, num_epochs=num_epochs)
        elif procedure == "PGDL2_FFT":
            atk = PGDL2_(model, eps=eps2, steps=20, transf=fft_transf, random_start=r_start)
            cost1=train_(model,train_loader,optim,loss,atk, num_epochs=num_epochs)
        elif procedure == "PGDL2_LOG":
            atk = PGDL2_(model, eps=eps2, steps=20, transf=log_transf, random_start=r_start)
            cost1=train_(model,train_loader,optim,loss,atk, num_epochs=num_epochs)

        
        
        print(f"Finish training, starting testing. Iter: {i+1}/{n_models}.")
        acc = get_model_acc(model, test_loader, atks_model_test)
        
        for (key, value) in acc.items():
            temp = iters_result[key]
            temp.append(value)
            iters_result[key] = temp
            
        print(f'Model average. Iter: {i+1}/{n_models}.')
            
    results = {}
    avg_all =[]
    
    for (key, values) in iters_result.items():
        aux_array = np.array(values)
        mean = np.mean(aux_array)
        std = np.std(aux_array)
        avg_all.append(mean)
        results[key] = str(round(mean, 2))+"+"+str(round(std, 2))
        
    avg_all = np.array(avg_all)
    mean_all = np.mean(avg_all)
    std_all = np.std(avg_all)
    results['AVG_ALL'] = str(round(mean_all, 2))+"+"+str(round(std_all, 2))
    
    return results

if __name__ == "__main__":
    
    std_acc = get_model_avg(n_models=5)
    std_acc['Name'] = 'STD'
    df = pd.DataFrame(columns = list(std_acc.keys()))
    df = df.append(std_acc, ignore_index=True)
    
    rr_acc = get_model_avg(n_models=5, procedure='RR')
    rr_acc['Name'] = 'RR'
    df = df.append(rr_acc, ignore_index=True)
    
    rr_acc = get_model_avg(n_models=5, procedure='Greedy')
    rr_acc['Name'] = 'Greedy'
    df = df.append(rr_acc, ignore_index=True)
    
    rr_acc = get_model_avg(n_models=5, procedure='MW')
    rr_acc['Name'] = 'MW'
    df = df.append(rr_acc, ignore_index=True)
    
    pd.set_option('display.max_columns', None)
    print(df)