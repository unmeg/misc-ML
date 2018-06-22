import torch
import torch.nn as nn 
import torch.backends.cudnn as cudnn 
import torch.optim as optim 
import numpy as np 

class SiameseTwinnies(nn.Module):
    def __init__(self):
        super(SiameseTwinnies, self).__init__()

        self.cnn = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(in_place=true),
            nn.BatchNorm2d(4),
            nn.Dropout(p=0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(in_place=true),
            nn.BatchNorm2d(8),
            nn.Dropout(p=0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(in_place=true),
            nn.BatchNorm2d(8),
            nn.Dropout(p=0.2),

        )

        self.fc = nn.Sequential(
            nn.Linear(8*100*100, 5),
            nn.ReLU(inplace=true),

            nn.Linear(500, 500),
            nn.ReLU(inplace=true),

            nn.Linear(500, 5), 
        )

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)

        return output

    def forward(self, in_one, in_two):
        output1 = forward_once(in_one)
        output2 = forward_once(in_two)

        return output1, output2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1-label) * torch.pow(euclidean_distance, 2) + 
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),2) # clamp makes everything sit between two values: in this case we just have one boundary, so everything will be greater than 0
        )

        return loss_contrastive