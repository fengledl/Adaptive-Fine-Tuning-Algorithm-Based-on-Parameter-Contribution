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
import warnings
temp = sys.stdout

f = open('code.log', 'w')

sys.stdout = f
warnings.filterwarnings("ignore")
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





data_name = ['indoor', 'stf_dog','aircraft', 'ucf101', 'omniglot','caltech256-30','caltech256-60']
num_classes = [67, 120, 100, 101, 1623,256,256]

seed = -1
for seed_ind in range(10):
    seed += 1
    for data_index in [0,1,2,3,4,5,6]:
        print(data_name[data_index])
        setup_seed(seed)
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        batch_size = 64
        data_dir="./datasets/"+data_name[data_index]+'/'


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



        for epoch in range(1, num_epochs + 1):
            start = time.time()
            print(f'Epoch {epoch}/{num_epochs}')
            pretrained_net.train()
            train_loss_list = []
            train_pre_list = []
            train_label_list = []
            ######
            if epoch%30==0:
                lr = lr * 0.1
            if epoch == 1:
                for images, labels in tqdm(train_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = pretrained_net(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    break

            RGN_w = []
            RGN_b = []
            for name, params in pretrained_net.named_parameters():
                if '.weight' in name:
                    para_flatten_norm = torch.norm(torch.flatten(params), p=2)
                    grad_flatten_norm = torch.norm(torch.flatten(params.grad), p=2)
                    RGN = grad_flatten_norm / para_flatten_norm
                    RGN_w.append(RGN)
                if '.bias' in name:
                    para_flatten_norm = torch.norm(torch.flatten(params), p=2)
                    grad_flatten_norm = torch.norm(torch.flatten(params.grad), p=2)
                    RGN = grad_flatten_norm / para_flatten_norm
                    RGN_b.append(RGN)
            RGN_w = [(x-min(RGN_w))/(max(RGN_w)-min(RGN_w)) for x in RGN_w]
            RGN_b = [(x - min(RGN_b)) / (max(RGN_b) - min(RGN_b)) for x in RGN_b]

            optimizer = optim.SGD([{'params': pretrained_net.conv1.weight, 'lr': lr * RGN_w[0]},
                                   {'params': pretrained_net.bn1.weight, 'lr': lr * RGN_w[1]},
                                   {'params': pretrained_net.layer1[0].conv1.weight, 'lr': lr * RGN_w[2]},
                                   {'params': pretrained_net.layer1[0].bn1.weight, 'lr': lr * RGN_w[3]},
                                   {'params': pretrained_net.layer1[0].conv2.weight, 'lr': lr * RGN_w[4]},
                                   {'params': pretrained_net.layer1[0].bn2.weight, 'lr': lr * RGN_w[5]},
                                   {'params': pretrained_net.layer1[0].conv3.weight, 'lr': lr * RGN_w[6]},
                                   {'params': pretrained_net.layer1[0].bn3.weight, 'lr': lr * RGN_w[7]},
                                   {'params': pretrained_net.layer1[0].downsample[0].weight, 'lr': lr * RGN_w[8]},
                                   {'params': pretrained_net.layer1[0].downsample[1].weight, 'lr': lr * RGN_w[9]},

                                   {'params': pretrained_net.layer1[1].conv1.weight, 'lr': lr * RGN_w[10]},
                                   {'params': pretrained_net.layer1[1].bn1.weight, 'lr': lr * RGN_w[11]},
                                   {'params': pretrained_net.layer1[1].conv2.weight, 'lr': lr * RGN_w[12]},
                                   {'params': pretrained_net.layer1[1].bn2.weight, 'lr': lr * RGN_w[13]},
                                   {'params': pretrained_net.layer1[1].conv3.weight, 'lr': lr * RGN_w[14]},
                                   {'params': pretrained_net.layer1[1].bn3.weight, 'lr': lr * RGN_w[15]},

                                   {'params': pretrained_net.layer1[2].conv1.weight, 'lr': lr * RGN_w[16]},
                                   {'params': pretrained_net.layer1[2].bn1.weight, 'lr': lr * RGN_w[17]},
                                   {'params': pretrained_net.layer1[2].conv2.weight, 'lr': lr * RGN_w[18]},
                                   {'params': pretrained_net.layer1[2].bn2.weight, 'lr': lr * RGN_w[19]},
                                   {'params': pretrained_net.layer1[2].conv3.weight, 'lr': lr * RGN_w[20]},
                                   {'params': pretrained_net.layer1[2].bn3.weight, 'lr': lr * RGN_w[21]},


                                   {'params': pretrained_net.layer2[0].conv1.weight, 'lr': lr * RGN_w[22]},
                                   {'params': pretrained_net.layer2[0].bn1.weight, 'lr': lr * RGN_w[23]},
                                   {'params': pretrained_net.layer2[0].conv2.weight, 'lr': lr * RGN_w[24]},
                                   {'params': pretrained_net.layer2[0].bn2.weight, 'lr': lr * RGN_w[25]},
                                   {'params': pretrained_net.layer2[0].conv3.weight, 'lr': lr * RGN_w[26]},
                                   {'params': pretrained_net.layer2[0].bn3.weight, 'lr': lr * RGN_w[27]},
                                   {'params': pretrained_net.layer2[0].downsample[0].weight, 'lr': lr * RGN_w[28]},
                                   {'params': pretrained_net.layer2[0].downsample[1].weight, 'lr': lr * RGN_w[29]},

                                   {'params': pretrained_net.layer2[1].conv1.weight, 'lr': lr * RGN_w[30]},
                                   {'params': pretrained_net.layer2[1].bn1.weight, 'lr': lr * RGN_w[31]},
                                   {'params': pretrained_net.layer2[1].conv2.weight, 'lr': lr * RGN_w[32]},
                                   {'params': pretrained_net.layer2[1].bn2.weight, 'lr': lr * RGN_w[33]},
                                   {'params': pretrained_net.layer2[1].conv3.weight, 'lr': lr * RGN_w[34]},
                                   {'params': pretrained_net.layer2[1].bn3.weight, 'lr': lr * RGN_w[35]},

                                   {'params': pretrained_net.layer2[2].conv1.weight, 'lr': lr * RGN_w[36]},
                                   {'params': pretrained_net.layer2[2].bn1.weight, 'lr': lr * RGN_w[37]},
                                   {'params': pretrained_net.layer2[2].conv2.weight, 'lr': lr * RGN_w[38]},
                                   {'params': pretrained_net.layer2[2].bn2.weight, 'lr': lr * RGN_w[39]},
                                   {'params': pretrained_net.layer2[2].conv3.weight, 'lr': lr * RGN_w[40]},
                                   {'params': pretrained_net.layer2[2].bn3.weight, 'lr': lr * RGN_w[41]},

                                   {'params': pretrained_net.layer2[3].conv1.weight, 'lr': lr * RGN_w[42]},
                                   {'params': pretrained_net.layer2[3].bn1.weight, 'lr': lr * RGN_w[43]},
                                   {'params': pretrained_net.layer2[3].conv2.weight, 'lr': lr * RGN_w[44]},
                                   {'params': pretrained_net.layer2[3].bn2.weight, 'lr': lr * RGN_w[45]},
                                   {'params': pretrained_net.layer2[3].conv3.weight, 'lr': lr * RGN_w[46]},
                                   {'params': pretrained_net.layer2[3].bn3.weight, 'lr': lr * RGN_w[47]},


                                   {'params': pretrained_net.layer3[0].conv1.weight, 'lr': lr * RGN_w[48]},
                                   {'params': pretrained_net.layer3[0].bn1.weight, 'lr': lr * RGN_w[49]},
                                   {'params': pretrained_net.layer3[0].conv2.weight, 'lr': lr * RGN_w[50]},
                                   {'params': pretrained_net.layer3[0].bn2.weight, 'lr': lr * RGN_w[51]},
                                   {'params': pretrained_net.layer3[0].conv3.weight, 'lr': lr * RGN_w[52]},
                                   {'params': pretrained_net.layer3[0].bn3.weight, 'lr': lr * RGN_w[53]},
                                   {'params': pretrained_net.layer3[0].downsample[0].weight, 'lr': lr * RGN_w[54]},
                                   {'params': pretrained_net.layer3[0].downsample[1].weight, 'lr': lr * RGN_w[55]},

                                   {'params': pretrained_net.layer3[1].conv1.weight, 'lr': lr * RGN_w[56]},
                                   {'params': pretrained_net.layer3[1].bn1.weight, 'lr': lr * RGN_w[57]},
                                   {'params': pretrained_net.layer3[1].conv2.weight, 'lr': lr * RGN_w[58]},
                                   {'params': pretrained_net.layer3[1].bn2.weight, 'lr': lr * RGN_w[59]},
                                   {'params': pretrained_net.layer3[1].conv3.weight, 'lr': lr * RGN_w[60]},
                                   {'params': pretrained_net.layer3[1].bn3.weight, 'lr': lr * RGN_w[61]},

                                   {'params': pretrained_net.layer3[2].conv1.weight, 'lr': lr * RGN_w[62]},
                                   {'params': pretrained_net.layer3[2].bn1.weight, 'lr': lr * RGN_w[63]},
                                   {'params': pretrained_net.layer3[2].conv2.weight, 'lr': lr * RGN_w[64]},
                                   {'params': pretrained_net.layer3[2].bn2.weight, 'lr': lr * RGN_w[65]},
                                   {'params': pretrained_net.layer3[2].conv3.weight, 'lr': lr * RGN_w[66]},
                                   {'params': pretrained_net.layer3[2].bn3.weight, 'lr': lr * RGN_w[67]},

                                   {'params': pretrained_net.layer3[3].conv1.weight, 'lr': lr * RGN_w[68]},
                                   {'params': pretrained_net.layer3[3].bn1.weight, 'lr': lr * RGN_w[69]},
                                   {'params': pretrained_net.layer3[3].conv2.weight, 'lr': lr * RGN_w[70]},
                                   {'params': pretrained_net.layer3[3].bn2.weight, 'lr': lr * RGN_w[71]},
                                   {'params': pretrained_net.layer3[3].conv3.weight, 'lr': lr * RGN_w[72]},
                                   {'params': pretrained_net.layer3[3].bn3.weight, 'lr': lr * RGN_w[73]},

                                   {'params': pretrained_net.layer3[4].conv1.weight, 'lr': lr * RGN_w[74]},
                                   {'params': pretrained_net.layer3[4].bn1.weight, 'lr': lr * RGN_w[75]},
                                   {'params': pretrained_net.layer3[4].conv2.weight, 'lr': lr * RGN_w[76]},
                                   {'params': pretrained_net.layer3[4].bn2.weight, 'lr': lr * RGN_w[77]},
                                   {'params': pretrained_net.layer3[4].conv3.weight, 'lr': lr * RGN_w[78]},
                                   {'params': pretrained_net.layer3[4].bn3.weight, 'lr': lr * RGN_w[79]},

                                   {'params': pretrained_net.layer3[5].conv1.weight, 'lr': lr * RGN_w[80]},
                                   {'params': pretrained_net.layer3[5].bn1.weight, 'lr': lr * RGN_w[81]},
                                   {'params': pretrained_net.layer3[5].conv2.weight, 'lr': lr * RGN_w[82]},
                                   {'params': pretrained_net.layer3[5].bn2.weight, 'lr': lr * RGN_w[83]},
                                   {'params': pretrained_net.layer3[5].conv3.weight, 'lr': lr * RGN_w[84]},
                                   {'params': pretrained_net.layer3[5].bn3.weight, 'lr': lr * RGN_w[85]},


                                   {'params': pretrained_net.layer4[0].conv1.weight, 'lr': lr * RGN_w[86]},
                                   {'params': pretrained_net.layer4[0].bn1.weight, 'lr': lr * RGN_w[87]},
                                   {'params': pretrained_net.layer4[0].conv2.weight, 'lr': lr * RGN_w[88]},
                                   {'params': pretrained_net.layer4[0].bn2.weight, 'lr': lr * RGN_w[89]},
                                   {'params': pretrained_net.layer4[0].conv3.weight, 'lr': lr * RGN_w[90]},
                                   {'params': pretrained_net.layer4[0].bn3.weight, 'lr': lr * RGN_w[91]},
                                   {'params': pretrained_net.layer4[0].downsample[0].weight, 'lr': lr * RGN_w[92]},
                                   {'params': pretrained_net.layer4[0].downsample[1].weight, 'lr': lr * RGN_w[93]},

                                   {'params': pretrained_net.layer4[1].conv1.weight, 'lr': lr * RGN_w[94]},
                                   {'params': pretrained_net.layer4[1].bn1.weight, 'lr': lr * RGN_w[95]},
                                   {'params': pretrained_net.layer4[1].conv2.weight, 'lr': lr * RGN_w[96]},
                                   {'params': pretrained_net.layer4[1].bn2.weight, 'lr': lr * RGN_w[97]},
                                   {'params': pretrained_net.layer4[1].conv3.weight, 'lr': lr * RGN_w[98]},
                                   {'params': pretrained_net.layer4[1].bn3.weight, 'lr': lr * RGN_w[99]},

                                   {'params': pretrained_net.layer4[2].conv1.weight, 'lr': lr * RGN_w[100]},
                                   {'params': pretrained_net.layer4[2].bn1.weight, 'lr': lr * RGN_w[101]},
                                   {'params': pretrained_net.layer4[2].conv2.weight, 'lr': lr * RGN_w[102]},
                                   {'params': pretrained_net.layer4[2].bn2.weight, 'lr': lr * RGN_w[103]},
                                   {'params': pretrained_net.layer4[2].conv3.weight, 'lr': lr * RGN_w[104]},
                                   {'params': pretrained_net.layer4[2].bn3.weight, 'lr': lr * RGN_w[105]},


                                   {'params': pretrained_net.fc.weight, 'lr': lr * RGN_w[106]},


                                   {'params': pretrained_net.bn1.bias, 'lr': lr * RGN_b[0]},

                                   {'params': pretrained_net.layer1[0].bn1.bias, 'lr': lr * RGN_b[1]},
                                   {'params': pretrained_net.layer1[0].bn2.bias, 'lr': lr * RGN_b[2]},
                                   {'params': pretrained_net.layer1[0].bn3.bias, 'lr': lr * RGN_b[3]},
                                   {'params': pretrained_net.layer1[0].downsample[1].bias, 'lr': lr * RGN_b[4]},

                                   {'params': pretrained_net.layer1[1].bn1.bias, 'lr': lr * RGN_b[5]},
                                   {'params': pretrained_net.layer1[1].bn2.bias, 'lr': lr * RGN_b[6]},
                                   {'params': pretrained_net.layer1[1].bn3.bias, 'lr': lr * RGN_b[7]},

                                   {'params': pretrained_net.layer1[2].bn1.bias, 'lr': lr * RGN_b[8]},
                                   {'params': pretrained_net.layer1[2].bn2.bias, 'lr': lr * RGN_b[9]},
                                   {'params': pretrained_net.layer1[2].bn3.bias, 'lr': lr * RGN_b[10]},


                                   {'params': pretrained_net.layer2[0].bn1.bias, 'lr': lr * RGN_b[11]},
                                   {'params': pretrained_net.layer2[0].bn2.bias, 'lr': lr * RGN_b[12]},
                                   {'params': pretrained_net.layer2[0].bn3.bias, 'lr': lr * RGN_b[13]},
                                   {'params': pretrained_net.layer2[0].downsample[1].bias, 'lr': lr * RGN_b[14]},

                                   {'params': pretrained_net.layer2[1].bn1.bias, 'lr': lr * RGN_b[15]},
                                   {'params': pretrained_net.layer2[1].bn2.bias, 'lr': lr * RGN_b[16]},
                                   {'params': pretrained_net.layer2[1].bn3.bias, 'lr': lr * RGN_b[17]},

                                   {'params': pretrained_net.layer2[2].bn1.bias, 'lr': lr * RGN_b[18]},
                                   {'params': pretrained_net.layer2[2].bn2.bias, 'lr': lr * RGN_b[19]},
                                   {'params': pretrained_net.layer2[2].bn3.bias, 'lr': lr * RGN_b[20]},

                                   {'params': pretrained_net.layer2[3].bn1.bias, 'lr': lr * RGN_b[21]},
                                   {'params': pretrained_net.layer2[3].bn2.bias, 'lr': lr * RGN_b[22]},
                                   {'params': pretrained_net.layer2[3].bn3.bias, 'lr': lr * RGN_b[23]},


                                   {'params': pretrained_net.layer3[0].bn1.bias, 'lr': lr * RGN_b[24]},
                                   {'params': pretrained_net.layer3[0].bn2.bias, 'lr': lr * RGN_b[25]},
                                   {'params': pretrained_net.layer3[0].bn3.bias, 'lr': lr * RGN_b[26]},
                                   {'params': pretrained_net.layer3[0].downsample[1].bias, 'lr': lr * RGN_b[27]},

                                   {'params': pretrained_net.layer3[1].bn1.bias, 'lr': lr * RGN_b[28]},
                                   {'params': pretrained_net.layer3[1].bn2.bias, 'lr': lr * RGN_b[29]},
                                   {'params': pretrained_net.layer3[1].bn3.bias, 'lr': lr * RGN_b[30]},

                                   {'params': pretrained_net.layer3[2].bn1.bias, 'lr': lr * RGN_b[31]},
                                   {'params': pretrained_net.layer3[2].bn2.bias, 'lr': lr * RGN_b[32]},
                                   {'params': pretrained_net.layer3[2].bn3.bias, 'lr': lr * RGN_b[33]},

                                   {'params': pretrained_net.layer3[3].bn1.bias, 'lr': lr * RGN_b[34]},
                                   {'params': pretrained_net.layer3[3].bn2.bias, 'lr': lr * RGN_b[35]},
                                   {'params': pretrained_net.layer3[3].bn3.bias, 'lr': lr * RGN_b[36]},

                                   {'params': pretrained_net.layer3[4].bn1.bias, 'lr': lr * RGN_b[37]},
                                   {'params': pretrained_net.layer3[4].bn2.bias, 'lr': lr * RGN_b[38]},
                                   {'params': pretrained_net.layer3[4].bn3.bias, 'lr': lr * RGN_b[39]},

                                   {'params': pretrained_net.layer3[5].bn1.bias, 'lr': lr * RGN_b[40]},
                                   {'params': pretrained_net.layer3[5].bn2.bias, 'lr': lr * RGN_b[41]},
                                   {'params': pretrained_net.layer3[5].bn3.bias, 'lr': lr * RGN_b[42]},

                                   {'params': pretrained_net.layer4[0].bn1.bias, 'lr': lr * RGN_b[43]},
                                   {'params': pretrained_net.layer4[0].bn2.bias, 'lr': lr * RGN_b[44]},
                                   {'params': pretrained_net.layer4[0].bn3.bias, 'lr': lr * RGN_b[45]},
                                   {'params': pretrained_net.layer4[0].downsample[1].bias, 'lr': lr * RGN_b[46]},

                                   {'params': pretrained_net.layer4[1].bn1.bias, 'lr': lr * RGN_b[47]},
                                   {'params': pretrained_net.layer4[1].bn2.bias, 'lr': lr * RGN_b[48]},
                                   {'params': pretrained_net.layer4[1].bn3.bias, 'lr': lr * RGN_b[49]},

                                   {'params': pretrained_net.layer4[2].bn1.bias, 'lr': lr * RGN_b[50]},
                                   {'params': pretrained_net.layer4[2].bn2.bias, 'lr': lr * RGN_b[51]},
                                   {'params': pretrained_net.layer4[2].bn3.bias, 'lr': lr * RGN_b[52]},


                                   {'params': pretrained_net.fc.bias, 'lr': lr * RGN_b[53]}
                                   ], momentum=0.9, weight_decay=0.0005)

            ######
            for images, labels in tqdm(train_loader):
                log_train = train_one_batch_finetune(images, labels, pretrained_net)
                train_loss_list.append(log_train['train_loss'])
                train_pre_list.extend(log_train['preds'])
                train_label_list.extend(log_train['labels'])
            train_loss = np.mean(train_loss_list)
            train_accuracy = accuracy_score(train_label_list, train_pre_list)
            pretrained_net.eval()
            log_test = evaluate_testset(pretrained_net)
            test_accuracy = log_test['test_accuracy']
            test_loss = log_test['test_loss']
            print(f'Epoch {epoch}/{num_epochs}, train_accuracy {train_accuracy}, loss {train_loss}, test_accuracy {test_accuracy}, loss {test_loss}, time {time.time() - start}')
        del pretrained_net
        gc.collect()

f.close()
