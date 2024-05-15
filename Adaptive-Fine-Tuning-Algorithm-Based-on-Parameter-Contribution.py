import random
import numpy as np
import time
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

# Save the output to a file
f = open('output.log', 'w')

# Then use the print function to print the content to a .log file
sys.stdout = f
warnings.filterwarnings("ignore")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_batch_finetune(images, labels, model, list_matrix, optimizer, criterion):
    '''
    Run a batch of training and return the training logs of the current batch
    :param images:
    :param labels:
    :return:
    '''
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()

    layer_names = []
    for name, value in model.named_parameters():
        if 'conv' in name:
            layer_names.append(name)

    layer_num = -1
    for i, (name, params) in enumerate(model.named_parameters()):
        if 'conv' in name:
            layer_num += 1
        if 'fc' not in name:
            matrix = list_matrix[i]
            matrix = matrix.cuda(0)
            params.grad[:] = params.grad * matrix

    optimizer.step()
    _,preds = torch.max(outputs, 1)
    preds = preds.cpu().numpy()
    loss = loss.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    log_train = {}
    log_train['train_loss'] = loss
    log_train['preds'] = preds
    log_train['labels'] = labels

    return log_train
def evaluate_testset(models, criterion):
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

    log_test['test_loss'] = np.mean(loss_list)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_precision'] = precision_score(labels_list, preds_list, average = 'macro')
    log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
    log_test['test_precision'] = f1_score(labels_list, preds_list, average='macro')

    return log_test


data_name = ['indoor', 'stf_dog', 'aircraft', 'ucf101', 'omniglot','caltech256-30','caltech256-60']
num_classes = [67, 120, 100, 101, 1623,256,256]
data_index = 0
print(data_name[data_index])

seed=0
setup_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
data_dir = "./datasets/" + data_name[data_index] + '/'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
train_loader = DataLoader(ImageFolder(os.path.join(data_dir, 'train'), transform=train_augs),
                          batch_size, shuffle=True)
test_loader = DataLoader(ImageFolder(os.path.join(data_dir, 'val'), transform=test_augs),
                         batch_size, shuffle=False, )


class PSO:

    def __init__(self, D, N, M, p_low, p_up, v_low, v_high, w, c1 = 1.5, c2 = 1.5):
        self.w = w  # Inertia weights
        self.c1 = c1  # Individual learning factors
        self.c2 = c2  # Population learning factor
        self.D = D  # Particle dimension
        self.N = N  # Population size, initialization of the number of populations
        self.M = M  # Maximum number of iterations
        self.p_range = [p_low, p_up]  # The extent of the constraint at the particle position
        self.v_range = [v_low, v_high]  # The range of constraints for particle velocity
        self.x = np.zeros((self.N, self.D))  # The position of all particles
        self.v = np.zeros((self.N, self.D))  # The velocity of all particles
        self.p_best = np.zeros((self.N, self.D))  # The optimal historical position of each particle
        self.g_best = np.zeros((1, self.D))[0]  # The optimal location of the population (global).
        self.p_bestFit = np.zeros(self.N)  # Initializes the optimal fit for each particle
        self.g_bestFit = 0.0000001#float('Inf')  # The optimal fitness value of the primary population (global).

        # Initialize all individual and global information
        for i in range(self.N):
            for j in range(self.D):
                self.x[i][j] = random.uniform(self.p_range[0], self.p_range[1])
                self.v[i][j] = random.uniform(self.v_range[0], self.v_range[1])
            self.p_best[i] = self.x[i]  
            fit = self.fitness(self.p_best[i])
            self.p_bestFit[i] = fit  
            if fit > self.g_bestFit: 
                self.g_best = self.p_best[i]
                self.g_bestFit = fit

    def fitness(self,x):
        setup_seed(seed)
        pretrained_net = models.resnet50(pretrained=True)
        # pretrained_net = models.resnet50(weights=None)
        # pathfile = 'pretrained_models/resnet50-0676ba61.pth'
        # pretrained_net.load_state_dict(torch.load(pathfile))
        pretrained_net.fc = nn.Linear(2048, num_classes[data_index])
        pretrained_net = pretrained_net.to(device)

        lr, num_epochs = 0.01, 5

        optimizer = optim.SGD(pretrained_net.parameters(), momentum=0.9, lr=lr, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss()

        list_matrix = []
        layer_num = -1
        layer_rates = x
        for name, params in pretrained_net.named_parameters():
            if 'conv' in name:
                layer_num += 1
            if 'fc' not in name:
                layer_rate = layer_rates[layer_num]
                flat_params = params.view(-1)
                sort, ind_flat_params = torch.sort(flat_params)
                flat_mat = torch.zeros_like(flat_params)
                flat_mat[ind_flat_params[:int(layer_rate * flat_mat.numel())]] = 1
                matrix = torch.reshape(flat_mat, params.shape)
                list_matrix.append(matrix)

        test_accuracy_list = []
        test_loss_list = []
        for epoch in range(1, num_epochs + 1):
            start = time.time()
            pretrained_net.train()
            train_loss_list = []
            train_pre_list = []
            train_label_list = []
            for images, labels in tqdm(train_loader):
                log_train = train_one_batch_finetune(images, labels, pretrained_net, list_matrix, optimizer, criterion)
                train_loss_list.append(log_train['train_loss'])
                train_pre_list.extend(log_train['preds'])
                train_label_list.extend(log_train['labels'])
            train_loss = np.mean(train_loss_list)
            train_accuracy = accuracy_score(train_label_list, train_pre_list)
            pretrained_net.eval()
            log_test = evaluate_testset(pretrained_net, criterion)
            test_accuracy = log_test['test_accuracy']
            test_loss = log_test['test_loss']
            test_accuracy_list.append(test_accuracy)
            test_loss_list.append(test_loss)
            print(
                f'Epoch {epoch}/{num_epochs}, train_accuracy {train_accuracy}, loss {train_loss}, test_accuracy {test_accuracy}, loss {test_loss}, time {time.time() - start}')
        del pretrained_net
        gc.collect()
        max_test_acc = max(test_accuracy_list)
        random.seed() #Unrandom seeds
        np.random.seed() #Unrandom seeds
        beta = 0.001
        fit = max_test_acc
        return fit


    def update(self):
        for i in range(self.N):
            # Update speed
            self.v[i] = self.w * self.v[i] + self.c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.x[i]) + self.c2 * random.uniform(0, 1) * (self.g_best - self.x[i])
            # Speed limit
            for j in range(self.D):
                if self.v[i][j] < self.v_range[0]:
                    self.v[i][j] = self.v_range[0]
                if self.v[i][j] > self.v_range[1]:
                    self.v[i][j] = self.v_range[1]
            # Update the location
            self.x[i] = self.x[i] + self.v[i]
            # Location limit
            for j in range(self.D):
                if self.x[i][j] < self.p_range[0]:
                    self.x[i][j] = self.p_range[0]
                if self.x[i][j] > self.p_range[1]:
                    self.x[i][j] = self.p_range[1]
            # Update individual and global historical optimal locations and adaptation values
            _fit = self.fitness(self.x[i])
            if _fit > self.p_bestFit[i]:
                self.p_best[i] = self.x[i].copy()
                self.p_bestFit[i] = _fit.copy()
            if _fit > self.g_bestFit:
                self.g_best = self.x[i].copy()
                self.g_bestFit = _fit.copy()

    def pso(self, draw = 1):
        best_fit = []  
        w_range = None
        if isinstance(self.w, tuple):
            w_range = self.w[1] - self.w[0]
            self.w = self.w[1]
        time_start = time.time()  
        for i in range(self.M):
            self.update()  # Update the main parameters and information
            if w_range:
                self.w -= w_range / self.M  # The inertia weight decreases linearly
            print("\rIter: {:d}/{:d} fitness: {:.4f} ".format(i, self.M, self.g_bestFit, end = '\n'))
            print('The best individuals is:', self.g_best,'\n')
            best_fit.append(self.g_bestFit.copy())
        time_end = time.time()  
        print(f'Algorithm takes {time_end - time_start} seconds') 
        return self.g_best



if __name__ == '__main__':
    low = 0
    up = 1
    pso = PSO(49, 10, 15, low, up, -0.2, 0.2, w = (0.7, 1.4))
    #pso(Dimensions, number of individuals, number of iterations, The minimum value of the position, The maximum value of the location, The minimum value of the velocity, The maximum value of the velocity, The range of inertia weights)
    g_best = pso.pso()



    setup_seed(seed) 
    print(data_name[data_index],'Final fine-tuning')
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

    list_matrix = []

    layer_num = -1
    layer_rates = g_best

    for name, params in pretrained_net.named_parameters():
        if 'conv' in name:
            layer_num += 1
        if 'fc' not in name:
            layer_rate = layer_rates[layer_num]
            flat_params = params.view(-1)
            sort, ind_flat_params = torch.sort(flat_params)
            flat_mat = torch.zeros_like(flat_params)
            flat_mat[ind_flat_params[:int(layer_rate*flat_mat.numel())]]=1
            matrix = torch.reshape(flat_mat, params.shape)
            list_matrix.append(matrix)


    for epoch in range(1, num_epochs + 1):
        start = time.time()
        print(f'Epoch {epoch}/{num_epochs}')
        pretrained_net.train()
        train_loss_list = []
        train_pre_list = []
        train_label_list = []
        for images, labels in tqdm(train_loader):
            log_train = train_one_batch_finetune(images, labels, pretrained_net, list_matrix, optimizer, criterion)
            train_loss_list.append(log_train['train_loss'])
            train_pre_list.extend(log_train['preds'])
            train_label_list.extend(log_train['labels'])
        train_loss = np.mean(train_loss_list)
        train_accuracy = accuracy_score(train_label_list, train_pre_list)
        lr_scheduler1.step()
        pretrained_net.eval()
        log_test = evaluate_testset(pretrained_net, criterion)
        test_accuracy = log_test['test_accuracy']
        test_loss = log_test['test_loss']
        print(f'Epoch {epoch}/{num_epochs}, train_accuracy {train_accuracy}, loss {train_loss}, test_accuracy {test_accuracy}, loss {test_loss}, time {time.time() - start}')

f.close()
