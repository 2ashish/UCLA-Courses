import torch

from torch import nn
import torch.nn.functional as F

class ModelRegressionAdt2Gex(nn.Module):
    def __init__(self, dim_mod1, dim_mod2):
        super(ModelRegressionAdt2Gex, self).__init__()
        self.input_ = nn.Linear(dim_mod1, 512)
        self.dropout1 = nn.Dropout(p=0.0)
        self.fc = nn.Linear(512, 512)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.output = nn.Linear(512, dim_mod2)
    def forward(self, x):
        x = F.gelu(self.input_(x))
        x = F.gelu(self.fc(x))
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.output(x))
        return x

class ConvNet4Adt2Gex(nn.Module):
    def __init__(self, dim_mod1, dim_mod2):
        super().__init__()
        self.conv1 = nn.Conv1d(1,8,3,padding='same')
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8,32,3,padding='same')
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32,64,3,padding='same')
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64,128,3,padding='same')
        self.bn4 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128*dim_mod1, 16*dim_mod1)
        self.fc1_bn = nn.BatchNorm1d(16*dim_mod1)
        self.dropout1 = nn.Dropout(p=0.0)
        self.fc2 = nn.Linear(16*dim_mod1, 16*dim_mod1)
        self.fc2_bn = nn.BatchNorm1d(16*dim_mod1)
        self.dropout2 = nn.Dropout(p=0.0)
        self.output = nn.Linear(16*dim_mod1, dim_mod2)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.output(x))
        #print(x.shape)
        return x
        
class ConvNet10Adt2Gex(nn.Module):
    def __init__(self, dim_mod1, dim_mod2):
        super().__init__()
        self.conv1 = nn.Conv1d(1,16,3,padding='same')
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16,64,3,padding='same')
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64,128,3,padding='same')
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128,128,3,padding='same')
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128,128,3,padding='same')
        self.bn5 = nn.BatchNorm1d(128)
        self.conv6 = nn.Conv1d(128,128,3,padding='same')
        self.bn6 = nn.BatchNorm1d(128)
        self.conv7 = nn.Conv1d(128,128,3,padding='same')
        self.bn7 = nn.BatchNorm1d(128)
        self.conv8 = nn.Conv1d(128,128,3,padding='same')
        self.bn8 = nn.BatchNorm1d(128)
        self.conv9 = nn.Conv1d(128,128,3,padding='same')
        self.bn9 = nn.BatchNorm1d(128)
        self.conv10 = nn.Conv1d(128,32,3,padding='same')
        self.bn10 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32*dim_mod1, 16*dim_mod1)
        self.fc1_bn = nn.BatchNorm1d(16*dim_mod1)
        #self.dropout1 = nn.Dropout(p=0.0)
        self.output = nn.Linear(16*dim_mod1, dim_mod2)
        
        
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = F.relu(self.bn3(self.conv3(x)))
        
        x = F.relu(self.bn4(self.conv4(x1)))
        x2 = F.relu(self.bn5(self.conv5(x)))
        
        x = x2+x1
        x = F.relu(self.bn6(self.conv6(x)))
        x3 = F.relu(self.bn7(self.conv7(x)))
        
        x = x3+x2
        x = F.relu(self.bn8(self.conv8(x)))
        x4 = F.relu(self.bn9(self.conv9(x)))
        
        x = x4+x3
        x = F.relu(self.bn10(self.conv10(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.output(x))
        
        return x