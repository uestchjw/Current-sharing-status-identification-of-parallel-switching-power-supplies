



from inspect import Parameter
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from hjw_Settings import begin,end,sub_len,sub_num,batch_size




''' 输入的data: 90*1000  
    输入的label:90*1,值只有0和1,其中1出现2次或3次'''


# a = torch.randn(size=(16,1000,90))
# print(a.size())
# c1,c2,c3,c4 = 1000,500,200,30
# conv1 = nn.Conv1d(c1,c2,5,1,2)
# b = conv1(a)
# print(b.size())
# bn1 = nn.BatchNorm1d(c2)
# c = bn1(b)
# print(c.size())
# relu = nn.ReLU(inplace=True)
# d = relu(c)
# print(d.size())

class CNN3_FC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        c1,c2,c3,c4 = 1000,500,200,30
        self.conv1 = nn.Conv1d(c1,c2,5,1,2)
        self.bn1 = nn.BatchNorm1d(c2)
        self.conv2 = nn.Conv1d(c2,c3,5,1,2)
        self.bn2 = nn.BatchNorm1d(c3)
        self.conv3 = nn.Conv1d(c3,c4,5,1,2)
        self.bn3 = nn.BatchNorm1d(c4)

        self.fc1 = nn.Linear(c4,10)
        self.fc2 = nn.Linear(10,3)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

        self.leakyrelu = nn.LeakyReLU(inplace=True)

    def forward(self,X):
        X = torch.permute(X,(0,2,1))
        out = self.relu(self.bn1(self.conv1(X)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))

        out = torch.permute(out,(0,2,1))
        out = self.relu(self.dropout(self.fc1(out)))

        out = self.fc2(out)
        return out


class CNN_LSTM1_1directional(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con = nn.Conv1d(1000,200,kernel_size=5,stride=1,padding=2)
        self.bn = nn.BatchNorm1d(200)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 200,hidden_size = 100,num_layers = 1,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*1,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*1,batch_size,100).float().to(self.device)

    def forward(self,X):
        # Dataset那里X是numpy类型
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # torch.Size([18, 1000, 90])
        out = self.con(X)            # torch.Size([18, 200, 90])
        
        out = self.bn(out)
        out = self.relu(out)
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = self.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out


class CNN_LSTM1_2directional(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con = nn.Conv1d(1000,200,kernel_size=5,stride=1,padding=2)
        self.bn = nn.BatchNorm1d(200)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 200,hidden_size = 100,num_layers = 1,batch_first = True,bidirectional = True)
        
        self.fc1 = nn.Linear(100*2,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(2*1,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(2*1,batch_size,100).float().to(self.device)

    def forward(self,X):
        # Dataset那里X是numpy类型
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # torch.Size([18, 1000, 90])
        out = self.con(X)            # torch.Size([18, 200, 90])
        
        out = self.bn(out)
        out = self.relu(out)
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = self.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class T1_CNN2_LSTM2_1directional(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,200,kernel_size=5,stride=1,padding=2)
        self.bn1 = nn.BatchNorm1d(200)

        self.conv2 = nn.Conv1d(200,30,5,1,2)
        self.bn2 = nn.BatchNorm1d(30)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 30,hidden_size = 15,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(15,5)
        self.fc2 = nn.Linear(5,2)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,15).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,15).float().to(self.device)

    def forward(self,X):
        # Dataset那里X是numpy类型
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # torch.Size([18, 1000, 90])
        out = self.con1(X)            # torch.Size([18, 200, 90])
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out


class T2_CNN2_LSTM2_1directional(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,300,kernel_size=5,stride=1,padding=2)
        self.bn1 = nn.BatchNorm1d(300)

        self.conv2 = nn.Conv1d(300,100,5,1,2)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,2)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # Dataset那里X是numpy类型
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # torch.Size([18, 1000, 90])
        out = self.con1(X)            # torch.Size([18, 200, 90])
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class T3_CNN2_LSTM2_1directional(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,300,kernel_size=5,stride=1,padding=2)
        self.bn1 = nn.BatchNorm1d(300)

        self.conv2 = nn.Conv1d(300,100,5,1,2)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 200,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(200,30)
        self.fc2 = nn.Linear(30,2)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,200).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,200).float().to(self.device)

    def forward(self,X):
        # Dataset那里X是numpy类型
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # torch.Size([18, 1000, 90])
        out = self.con1(X)            # torch.Size([18, 200, 90])
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class New1(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,600,kernel_size=5,stride=1,padding=2)
        self.con2 = nn.Conv1d(600,300,kernel_size=5,stride=1,padding=2)
        self.con2 = nn.Conv1d(300,100,kernel_size=5,stride=1,padding=2)
        self.bn1 = nn.BatchNorm1d(600)
        self.bn2 = nn.BatchNorm1d(300)
        self.bn3 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 3,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,2)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*3,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*3,batch_size,100).float().to(self.device)

    def forward(self,X):
        # Dataset那里X是numpy类型
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # torch.Size([18, 1000, 90])
                # torch.Size([18, 200, 90])

        out = self.relu(self.bn1(self.con1(X)))
        out = self.relu(self.bn2(self.con2(out)))
        out = self.relu(self.bn3(self.con3(out)))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out





''' 1层CNN,卷积核(感受野)不同 '''
class CNN_depth1_height1_1000_100_LSTM2(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con = nn.Conv1d(1000,100,kernel_size=1,stride=1,padding=0)
        self.bn = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # Dataset那里X是numpy类型
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # torch.Size([18, 1000, 90])
        out = self.con(X)            # torch.Size([18, 200, 90])
        
        out = self.bn(out)
        out = self.relu(out)
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth1_height3_1000_100_LSTM2(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con = nn.Conv1d(1000,100,kernel_size=3,stride=1,padding=1)
        self.bn = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # Dataset那里X是numpy类型
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # torch.Size([18, 1000, 90])
        out = self.con(X)            # torch.Size([18, 200, 90])
        
        out = self.bn(out)
        out = self.relu(out)
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth1_height5_1000_100_LSTM2(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con = nn.Conv1d(1000,100,kernel_size=5,stride=1,padding=2)
        self.bn = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # Dataset那里X是numpy类型
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # torch.Size([18, 1000, 90])
        out = self.con(X)            # torch.Size([18, 200, 90])
        
        out = self.bn(out)
        out = self.relu(out)
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth1_height7_1000_100_LSTM2(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con = nn.Conv1d(1000,100,kernel_size=7,stride=1,padding=3)
        self.bn = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # Dataset那里X是numpy类型
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # torch.Size([18, 1000, 90])
        out = self.con(X)            # torch.Size([18, 200, 90])
        
        out = self.bn(out)
        out = self.relu(out)
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth1_height9_1000_100_LSTM2(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con = nn.Conv1d(1000,100,kernel_size=9,stride=1,padding=4)
        self.bn = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # Dataset那里X是numpy类型
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # torch.Size([18, 1000, 90])
        out = self.con(X)            # torch.Size([18, 200, 90])
        
        out = self.bn(out)
        out = self.relu(out)
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth1_height11_1000_100_LSTM2(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con = nn.Conv1d(1000,100,kernel_size=11,stride=1,padding=5)
        self.bn = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # Dataset那里X是numpy类型
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # torch.Size([18, 1000, 90])
        out = self.con(X)            # torch.Size([18, 200, 90])
        
        out = self.bn(out)
        out = self.relu(out)
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth1_height13_1000_100_LSTM2(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con = nn.Conv1d(1000,100,kernel_size=13,stride=1,padding=6)
        self.bn = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # Dataset那里X是numpy类型
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # torch.Size([18, 1000, 90])
        out = self.con(X)            # torch.Size([18, 200, 90])
        
        out = self.bn(out)
        out = self.relu(out)
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth1_height15_1000_100_LSTM2(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con = nn.Conv1d(1000,100,kernel_size=15,stride=1,padding=7)
        self.bn = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # Dataset那里X是numpy类型
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # torch.Size([18, 1000, 90])
        out = self.con(X)            # torch.Size([18, 200, 90])
        
        out = self.bn(out)
        out = self.relu(out)
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out


''' 2层CNN,感受野为5 '''
class CNN_depth2_height5_300_100_LSTM2(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,300,kernel_size=5,stride=1,padding=2)
        self.bn1 = nn.BatchNorm1d(300)
        self.con2 = nn.Conv1d(300,100,kernel_size=5,stride=1,padding=2)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # [18,90,1000] -> [18, 1000, 90]
        out = self.con1(X)
        out = self.relu(self.bn1(out))
        out = self.con2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth2_height5_300_50_LSTM2(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,300,kernel_size=5,stride=1,padding=2)
        self.bn1 = nn.BatchNorm1d(300)
        self.con2 = nn.Conv1d(300,50,kernel_size=5,stride=1,padding=2)
        self.bn2 = nn.BatchNorm1d(50)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 50,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # [18,90,1000] -> [18, 1000, 90]
        out = self.con1(X)
        out = self.relu(self.bn1(out))
        out = self.con2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth2_height5_600_100_LSTM2(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,600,kernel_size=5,stride=1,padding=2)
        self.bn1 = nn.BatchNorm1d(600)
        self.con2 = nn.Conv1d(600,100,kernel_size=5,stride=1,padding=2)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # [18,90,1000] -> [18, 1000, 90]
        out = self.con1(X)
        out = self.relu(self.bn1(out))
        out = self.con2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth2_height5_600_300_LSTM2(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,600,kernel_size=5,stride=1,padding=2)
        self.bn1 = nn.BatchNorm1d(600)
        self.con2 = nn.Conv1d(600,300,kernel_size=5,stride=1,padding=2)
        self.bn2 = nn.BatchNorm1d(300)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 300,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # [18,90,1000] -> [18, 1000, 90]
        out = self.con1(X)
        out = self.relu(self.bn1(out))
        out = self.con2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out






''' 2层CNN,感受野为1,3,5,7 '''
class CNN_depth2_height1_300_100_LSTM2(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,300,kernel_size=1,stride=1,padding=0)
        self.bn1 = nn.BatchNorm1d(300)
        self.con2 = nn.Conv1d(300,100,kernel_size=1,stride=1,padding=0)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # [18,90,1000] -> [18, 1000, 90]
        out = self.con1(X)
        out = self.relu(self.bn1(out))
        out = self.con2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth2_height3_300_100_LSTM2(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,300,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm1d(300)
        self.con2 = nn.Conv1d(300,100,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # [18,90,1000] -> [18, 1000, 90]
        out = self.con1(X)
        out = self.relu(self.bn1(out))
        out = self.con2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth2_height5_300_100_LSTM2(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,300,kernel_size=5,stride=1,padding=2)
        self.bn1 = nn.BatchNorm1d(300)
        self.con2 = nn.Conv1d(300,100,kernel_size=5,stride=1,padding=2)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # [18,90,1000] -> [18, 1000, 90]
        out = self.con1(X)
        out = self.relu(self.bn1(out))
        out = self.con2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth2_height7_300_100_LSTM2(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,300,kernel_size=7,stride=1,padding=3)
        self.bn1 = nn.BatchNorm1d(300)
        self.con2 = nn.Conv1d(300,100,kernel_size=7,stride=1,padding=3)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # [18,90,1000] -> [18, 1000, 90]
        out = self.con1(X)
        out = self.relu(self.bn1(out))
        out = self.con2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out



''' 2层CNN, 感受野为3, LSTM层数不同 '''
class CNN_depth2_height3_300_100_LSTM1(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,300,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm1d(300)
        self.con2 = nn.Conv1d(300,100,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 1,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*1,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*1,batch_size,100).float().to(self.device)

    def forward(self,X):
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # [18,90,1000] -> [18, 1000, 90]
        out = self.con1(X)
        out = self.relu(self.bn1(out))
        out = self.con2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth2_height3_300_100_LSTM2(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,300,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm1d(300)
        self.con2 = nn.Conv1d(300,100,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # [18,90,1000] -> [18, 1000, 90]
        out = self.con1(X)
        out = self.relu(self.bn1(out))
        out = self.con2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth2_height3_300_100_LSTM3(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,300,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm1d(300)
        self.con2 = nn.Conv1d(300,100,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 3,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*3,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*3,batch_size,100).float().to(self.device)

    def forward(self,X):
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # [18,90,1000] -> [18, 1000, 90]
        out = self.con1(X)
        out = self.relu(self.bn1(out))
        out = self.con2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth2_height3_300_100_LSTM4(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,300,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm1d(300)
        self.con2 = nn.Conv1d(300,100,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 4,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*4,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*4,batch_size,100).float().to(self.device)

    def forward(self,X):
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # [18,90,1000] -> [18, 1000, 90]
        out = self.con1(X)
        out = self.relu(self.bn1(out))
        out = self.con2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth2_height3_300_100_LSTM5(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,300,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm1d(300)
        self.con2 = nn.Conv1d(300,100,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 5,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*5,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*5,batch_size,100).float().to(self.device)

    def forward(self,X):
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # [18,90,1000] -> [18, 1000, 90]
        out = self.con1(X)
        out = self.relu(self.bn1(out))
        out = self.con2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth2_height3_300_100_LSTM6(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,300,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm1d(300)
        self.con2 = nn.Conv1d(300,100,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 6,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*6,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*6,batch_size,100).float().to(self.device)

    def forward(self,X):
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # [18,90,1000] -> [18, 1000, 90]
        out = self.con1(X)
        out = self.relu(self.bn1(out))
        out = self.con2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out




''' LSTM隐藏层维度不同 '''
class CNN_depth2_height3_300_100_LSTM2_hidden50(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,300,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm1d(300)
        self.con2 = nn.Conv1d(300,100,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 50,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(50,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,50).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,50).float().to(self.device)

    def forward(self,X):
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # [18,90,1000] -> [18, 1000, 90]
        out = self.con1(X)
        out = self.relu(self.bn1(out))
        out = self.con2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth2_height3_300_100_LSTM2_hidden100(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,300,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm1d(300)
        self.con2 = nn.Conv1d(300,100,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # [18,90,1000] -> [18, 1000, 90]
        out = self.con1(X)
        out = self.relu(self.bn1(out))
        out = self.con2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth2_height3_300_100_LSTM2_hidden200(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,300,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm1d(300)
        self.con2 = nn.Conv1d(300,100,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 200,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(200,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,200).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,200).float().to(self.device)

    def forward(self,X):
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # [18,90,1000] -> [18, 1000, 90]
        out = self.con1(X)
        out = self.relu(self.bn1(out))
        out = self.con2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out




''' 最优模型尝试 '''
class CNN_depth2_height1_1_300_100_LSTM2(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,300,kernel_size=1,stride=1,padding=0)
        self.bn1 = nn.BatchNorm1d(300)
        self.con2 = nn.Conv1d(300,100,kernel_size=1,stride=1,padding=0)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # [18,90,1000] -> [18, 1000, 90]
        out = self.con1(X)
        out = self.relu(self.bn1(out))
        out = self.con2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth2_height1_3_300_100_LSTM2(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,300,kernel_size=1,stride=1,padding=0)
        self.bn1 = nn.BatchNorm1d(300)
        self.con2 = nn.Conv1d(300,100,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # [18,90,1000] -> [18, 1000, 90]
        out = self.con1(X)
        out = self.relu(self.bn1(out))
        out = self.con2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth2_height3_3_300_100_LSTM2(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,300,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm1d(300)
        self.con2 = nn.Conv1d(300,100,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # [18,90,1000] -> [18, 1000, 90]
        out = self.con1(X)
        out = self.relu(self.bn1(out))
        out = self.con2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth2_height3_5_300_100_LSTM2(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,300,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm1d(300)
        self.con2 = nn.Conv1d(300,100,kernel_size=5,stride=1,padding=2)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # [18,90,1000] -> [18, 1000, 90]
        out = self.con1(X)
        out = self.relu(self.bn1(out))
        out = self.con2(out)
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

class CNN_depth2_height1_3_3_5_300_100_LSTM2(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con1 = nn.Conv1d(1000,300,kernel_size=1,stride=1,padding=0)
        self.bn1 = nn.BatchNorm1d(300)
        self.con1_ = nn.Conv1d(1000,300,kernel_size=3,stride=1,padding=1)
        self.bn1_ = nn.BatchNorm1d(300)

        self.con2 = nn.Conv1d(300,100,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm1d(100)
        self.con2_ = nn.Conv1d(300,100,kernel_size=5,stride=1,padding=2)
        self.bn2_ = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 100,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # [18,90,1000] -> [18, 1000, 90]
        out1 = self.con1(X)
        out2 = self.con1_(X)
        out = (out1+out2)/2
        out = self.relu(self.bn1(out))
        out3 = self.con2(out)
        out4 = self.con2_(out)
        out = (out3+out4)/2
        out = self.relu(self.bn2(out))
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out






class CNN_LSTM2_1directional(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con = nn.Conv1d(1000,200,kernel_size=5,stride=1,padding=2)
        self.bn = nn.BatchNorm1d(200)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 200,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = False)
        
        self.fc1 = nn.Linear(100,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(1*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(1*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # Dataset那里X是numpy类型
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # torch.Size([18, 1000, 90])
        out = self.con(X)            # torch.Size([18, 200, 90])
        
        out = self.bn(out)
        out = self.relu(out)
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out



class hjw_onlyLSTM(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()

        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 1000,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = True)
        
        self.fc1 = nn.Linear(100*2,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(2*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(2*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # Dataset那里X是numpy类型
        # 输入的X是[18,90,1000],现在要对第三维进行卷积

        out,(h_n,c_n) = self.lstm(X,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out


''' 这个结构的Model,参数量 = Parameters: 1489892 trainable, 1489892 total'''
class hjw_PowerModel(nn.Module):
    def __init__(self,batch_size) -> None:
        super().__init__()
        self.con = nn.Conv1d(1000,200,kernel_size=5,stride=1,padding=2)
        self.bn = nn.BatchNorm1d(200)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = 200,hidden_size = 100,num_layers = 2,batch_first = True,bidirectional = True)
        
        self.fc1 = nn.Linear(100*2,30)
        self.fc2 = nn.Linear(30,3)
        self.dropout = nn.Dropout(p=0.5)
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = torch.rand(2*2,batch_size,100).float().to(self.device)
        self.c_0 = torch.rand(2*2,batch_size,100).float().to(self.device)

    def forward(self,X):
        # Dataset那里X是numpy类型
        # 输入的X是[18,90,1000],现在要对第三维进行卷积
        X = torch.permute(X,(0,2,1)) # torch.Size([18, 1000, 90])
        out = self.con(X)            # torch.Size([18, 200, 90])
        
        out = self.bn(out)
        out = self.relu(out)
        out = torch.permute(out,(0,2,1)) 
        out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        # out = F.softmax(out)
        return out

# class hjw_PowerModel(nn.Module):
#     def __init__(self,batch_size) -> None:
#         super().__init__()
#         self.con = nn.Conv1d(1000,50,kernel_size=5,stride=1,padding=2)
#         self.bn = nn.BatchNorm1d(50)
#         self.relu = nn.ReLU()
#         self.lstm = nn.LSTM(input_size = 50,hidden_size = 20,num_layers = 2,batch_first = True,bidirectional = True)
        
#         self.fc1 = nn.Linear(20*2,10)
#         self.fc2 = nn.Linear(10,2)
#         self.dropout = nn.Dropout(p=0.5)

#         self.h_0 = torch.rand(2*2,batch_size,20).float()
#         self.c_0 = torch.rand(2*2,batch_size,20).float()

#     def forward(self,X):
#         # Dataset那里X是numpy类型
#         # 输入的X是[18,90,1000],现在要对第三维进行卷积
#         X = torch.permute(X,(0,2,1)) # torch.Size([18, 1000, 90])
#         out = self.con(X)            # torch.Size([18, 200, 90])
        
#         out = self.bn(out)
#         out = self.relu(out)
#         out = torch.permute(out,(0,2,1)) 
#         out,(h_n,c_n) = self.lstm(out,(self.h_0,self.c_0))
#         out = F.relu(self.dropout(self.fc1(out)))
#         out = self.fc2(out)
#         # out = F.softmax(out)
#         return out


