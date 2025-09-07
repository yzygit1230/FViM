from __future__ import print_function
import argparse
import torch
from torch.nn import functional as F
from torchvision import transforms
import random
import numpy as np
import os
import warnings
import logging
import torch.nn as nn
logging.getLogger("profile").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")
from dataloader_ip import IntenPhaDataset
from torch.utils.data import DataLoader
from model.fvim import FViM

parser = argparse.ArgumentParser(description='Cell death')
parser.add_argument('--batchSize', type=int, default=1, metavar='N',     
                    help='input batch size for training (default: 128)')
parser.add_argument('--imageSize', type=int, default=256, metavar='N')    
parser.add_argument('--epochs', type=int, default=1000, metavar='N')    
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')                
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')    
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--seed', type=int, default=123, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()

def set_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
   
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
                 
set_random_seed(args.seed)          
train_on_gpu = torch.cuda.is_available()

device = torch.device("cuda:0" if train_on_gpu else "cpu")

data_transform1 = transforms.Compose([
    transforms.Resize((args.imageSize, args.imageSize)),
    transforms.ToTensor()
])
test_dataset = IntenPhaDataset(root_dir='celldeath/test', transform=data_transform1)
test_loader = DataLoader(test_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4)
pth_path = 'xx.pth'

def test():
    with torch.no_grad():
        torch.cuda.empty_cache()
        model.eval()
        test_acc = 0.0
        total = 0
        tmp_prediction = []
        tmp_scores = []
        total_label = []
        total_features = []
        all_predictions = []
        all_labels = []
        all_probs = []  

        TP = np.zeros(3)
        FP = np.zeros(3)
        FN = np.zeros(3)
        TN = np.zeros(3)

        for iteration, (img_i, img_p, labels, path_i, path_p) in enumerate(test_loader):
            img_i = img_i.to(device)
            img_p = img_p.to(device)
            images = torch.cat([img_i, img_p], 1)
            labels = labels.to(device)
    
            outputs, feature = model(img_i)
            outputs = F.softmax(outputs, dim=1)  
            total += labels.size(0)
            _, prediction = torch.max(outputs.data, 1)
            test_acc += torch.sum(prediction == labels)
            
            prediction = prediction.cpu().data.numpy()
            labels = labels.cpu().data.numpy()

            tmp_prediction.append(prediction)
            tmp_scores.append(outputs[0].data.cpu().numpy())
            total_label.append(labels)
            feature = feature.contiguous().view(images.size(0), -1)
            feature = feature.data.cpu().numpy()
            total_features.extend(feature)

            all_predictions.extend(prediction)
            all_labels.extend(labels)
            all_probs.append(outputs.cpu().data.numpy()) 

            for i in range(3): 
                TP[i] += np.sum((prediction == i) & (labels == i))  # True Positives
                FP[i] += np.sum((prediction == i) & (labels != i))  # False Positives
                FN[i] += np.sum((prediction != i) & (labels == i))  # False Negatives
                TN[i] += np.sum((prediction != i) & (labels != i))  # True Negatives

        precision = TP / (TP + FP)  
        recall = TP / (TP + FN)  
        p0 = (TP+TN) / (TP+FP+TN+FN)
        pe = ((TP+FP)*(TP+FN)+(FP+TN)+(FN+TN)) / (TP+FP+TN+FN)**2
        kappa = (p0-pe) / (1-pe)

        precision = np.mean(np.array(precision))
        recall = np.mean(np.array(recall))
        f1 = 2 * (precision * recall) / (precision + recall)

        mean_acc = test_acc / total
        print("Mean_Test_Acc:{:.4f}, Precision:{:.4f}, Recall:{:.4f}, F1:{:.4f}, Kappa:{:.4f}".format(mean_acc, precision, recall, f1, kappa))

        
if __name__ == '__main__' :
    model = FViM(num_classes=3)
    num_ftrs = model.cls_head.cls.in_features
    model.cls_head.cls = nn.Linear(num_ftrs, 3)
    model = model.to(device)
    model.load_state_dict(torch.load(pth_path, map_location={'cuda:0':'cuda:0'}))
    test()
