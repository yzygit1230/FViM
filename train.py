from __future__ import print_function
import argparse
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from dataloader_ip import IntenPhaDataset
import os
from torch.utils.data import DataLoader
from torchtoolbox.tools import mixup_data, mixup_criterion
import random
import numpy as np
from tqdm import tqdm
import os
from model.fvim import FViM

ckp_path = 'checkpoint'
if not os.path.exists(ckp_path):
    os.makedirs(ckp_path)

parser = argparse.ArgumentParser(description='Cell death')
parser.add_argument('--batchSize', type=int, default=64, metavar='N',     
                    help='input batch size for training (default: 128)')
parser.add_argument('--imageSize', type=int, default=256, metavar='N')    
parser.add_argument('--epochs', type=int, default=100, metavar='N')   
parser.add_argument('--lr', type=float, default=0.00008, help='Learning Rate. Default=0.002')             
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--seed', type=int, default=123, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
print(args)   
            
train_on_gpu = torch.cuda.is_available()

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

device = torch.device("cuda:0" if train_on_gpu else "cpu")
data_transform = transforms.Compose([
    transforms.Resize((args.imageSize, args.imageSize)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
data_transform1 = transforms.Compose([
    transforms.Resize((args.imageSize, args.imageSize)),
    transforms.ToTensor()
])

train_dataset = IntenPhaDataset(root_dir='celldeath/train', transform=data_transform)
val_dataset = IntenPhaDataset(root_dir='celldeath/val', transform=data_transform1)

train_loader = DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4)

def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p*F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)

model = FViM(num_classes=3)
num_ftrs = model.cls_head.cls.in_features
model.cls_head.cls = nn.Linear(num_ftrs, 3)

criterion = nn.CrossEntropyLoss()
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True,
                                                             threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-06)

alpha=0.3
def train(epoch):
    model.train()
    train_loss = 0
    train_acc = 0.0
    train_loss = 0.0
    total =0
    train_iterator = tqdm(train_loader,leave=True, total=len(train_loader),position=0,ncols=10)
    iterators=0
    for img_i, img_p, labels, path_i,path_p in train_iterator:
        iterators=iterators+1
        img_i = img_i.to(device)
        img_p = img_p.to(device)
        images = torch.cat([img_i,img_p], 1)
        labels=labels.to(device)
        optimizer.zero_grad()
        data, labels_a, labels_b, lam = mixup_data(images, labels, alpha)
 
        outputs, feature = model(images)
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        loss += entropy_loss(outputs)


        train_loss += loss.item()
        total += labels.size(0)
     
        _, prediction  = torch.max(outputs.data, 1)
        train_acc += torch.sum(prediction == labels)
        status="===> Epoch[{}]({}/{}): train_loss:{:.4f},mean_loss:{:.4f}, train_acc:{:.4f}".format(
        epoch, iterators, len(train_loader), loss.item(),train_loss/total,train_acc/total)
        loss.backward()
        optimizer.step()

    print(status)

    return train_loss/total,train_acc/total

def test(epoch):
    with torch.no_grad():
        torch.cuda.empty_cache()
        model.eval()
        test_acc = 0.0
        test_acc0 = 0.0
        test_acc1 = 0.0
        test_acc2 = 0.0

        total =0
        total0 =0
        total1 =0
        total2 =0
        
        for iteration, (img_i, img_p, labels, path_i, path_p) in enumerate(val_loader):
            img_i = img_i.to(device)
            img_p = img_p.to(device)
            images = torch.cat([img_i,img_p], 1)
            labels=labels.to(device)
            outputs, feature = model(images)
            outputs=F.softmax(outputs)         
            total += labels.size(0)
           
            _, prediction  = torch.max(outputs.data, 1)
            test_acc += torch.sum(prediction == labels)
            index0=(labels == 0).nonzero()
            total0 += index0.size(0)
            test_acc0 += torch.sum(prediction[index0] == labels[index0])
                
            index1=(labels == 1).nonzero()
            total1 += index1.size(0)
            test_acc1 += torch.sum(prediction[index1] == labels[index1])
            
            index2=(labels == 2).nonzero()
            total2 += index2.size(0)
            test_acc2 += torch.sum(prediction[index2] == labels[index2])
            

    print("===> Epoch[{}] =====>Mean_Test_Acc:{:.4f},Acc0:{:.4f},Acc1:{:.4f},Acc2:{:.4f}".format(
            epoch, test_acc/total,test_acc0/total0,test_acc1/total1,test_acc2/total2))
       
    return test_acc/total

def checkpoint(name='checkpoint', epoch=1, test_acc=1):
    torch.save(model.state_dict(), ckp_path+"/E:{}_ACC:{:.4f}.pth".format(epoch, test_acc))
    print("\n===>Checkpoint saved to "+ckp_path+"/E:{}_ACC:{:.4f}.pth".format(epoch, test_acc))

if __name__ == '__main__' :

    import os
    import matplotlib
    matplotlib.use('agg')

    log_dir='log'
   
    test_acc_list=[]
    train_acc_list=[]
    train_loss_list=[]

    for epoch in range(1, args.epochs + 1): 
        
        train_loss,train_acc=train(epoch)
        test_acc=test(epoch)
        test_acc_list.append(test_acc.cpu().data.numpy())
        train_acc_list.append(train_acc.cpu().data.numpy())
        train_loss_list.append(train_loss) 
        val_acc=test_acc

        if epoch==1:
            checkpoint(log_dir, epoch, test_acc)
            test_best_acc=test_acc
            val_best_acc=  val_acc

        if val_acc > val_best_acc:
           checkpoint(log_dir, epoch, test_acc)
           test_best_acc= test_acc
           val_best_acc= val_acc
            

        print("\n===>train_acc not be improved  to {:.4f}".format(train_acc))
        print("\n===>test_acc not be improved  to {:.4f}".format( test_best_acc))
           
