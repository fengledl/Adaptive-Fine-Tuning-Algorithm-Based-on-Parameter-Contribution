from torchvision.datasets import ImageFolder
import sys
sys.path.append('..')
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import random
import time
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import gc
temp = sys.stdout

f = open('code.log', 'w')

sys.stdout = f

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_batch_finetune(images, labels, model):
    '''
    :param images:
    :param labels:
    :return:
    '''
    #model = torch.nn.DataParallel(model)
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    _,preds = torch.max(outputs, 1)
    preds = preds.cpu().numpy()
    loss = loss.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    log_train = {}
    log_train['epoch'] = epoch
    log_train['train_loss'] = loss
    log_train['preds'] = preds
    log_train['labels'] = labels

    return log_train
def evaluate_testset(models):
    loss_list = []
    labels_list = []
    preds_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = models(images)

            _,preds = torch.max(outputs,1)
            preds = preds.cpu().numpy()
            loss = criterion(outputs, labels)
            loss = loss.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)
    log_test = {}
    log_test['epoch'] = epoch

    log_test['test_loss'] = np.mean(loss_list)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_precision'] = precision_score(labels_list, preds_list, average = 'macro')
    log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
    log_test['test_precision'] = f1_score(labels_list, preds_list, average='macro')

    return log_test





data_name = ['indoor', 'stf_dog', 'aircraft', 'ucf101', 'omniglot','caltech256-30','caltech256-60']
num_classes = [67, 120, 100, 101, 1623,256,256]
seed = -1
for iteration in range(10):
    seed += 1
    for data_index in [0,1,2,3,4,5,6]:
        group_param = group_params[0]
        print(data_name[data_index])
        setup_seed(seed)
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        batch_size = 64
        data_dir = "./datasets/" + data_name[data_index] + '/'


        normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        train_augs = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        test_augs = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            normalize
        ])

        train_loader = DataLoader(ImageFolder(os.path.join(data_dir,'train'),transform=train_augs),
                                batch_size,shuffle=True)
        test_loader = DataLoader(ImageFolder(os.path.join(data_dir,'val'),transform=test_augs),
                                batch_size,shuffle=False,)


        pretrained_net = models.resnet50(pretrained=True)
        # pretrained_net = models.resnet50(weights=None)
        # pathfile = 'pretrained_models/resnet50-0676ba61.pth'
        # pretrained_net.load_state_dict(torch.load(pathfile))
        pretrained_net.fc = nn.Linear(2048,num_classes[data_index])
        pretrained_net = pretrained_net.to(device)
        lr,num_epochs = 0.01,110

        optimizer = optim.SGD(pretrained_net.parameters(),momentum=0.9,lr=lr,weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss()
        lr_scheduler1 = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


        for epoch in range(1, num_epochs + 1):
            start = time.time()
            pretrained_net.train()
            train_loss_list = []
            train_pre_list = []
            train_label_list = []
            for images, labels in tqdm(train_loader):
                log_train = train_one_batch_finetune(images, labels, pretrained_net)
                train_loss_list.append(log_train['train_loss'])
                train_pre_list.extend(log_train['preds'])
                train_label_list.extend(log_train['labels'])
            train_loss = np.mean(train_loss_list)
            train_accuracy = accuracy_score(train_label_list, train_pre_list)
            lr_scheduler1.step()
            pretrained_net.eval()
            log_test = evaluate_testset(pretrained_net)
            test_accuracy = log_test['test_accuracy']
            test_loss = log_test['test_loss']
            print(f'Epoch {epoch}/{num_epochs}, train_accuracy {train_accuracy}, loss {train_loss}, test_accuracy {test_accuracy}, loss {test_loss}, time {time.time() - start}')
        del pretrained_net
        del lr_scheduler1
        gc.collect()
f.close()
