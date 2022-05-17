





import torch
import torch.nn as nn
from torch import optim
import numpy as np
from hjw_Settings import begin,end,sub_len,sub_num,batch_size
from hjw_PowerDataset import hjw_ThreeSMPS_DataLoader, hjw_TwoSMPS_DataLoader
from hjw_PowerModel import CNN3_FC, T1_CNN2_LSTM2_1directional,T2_CNN2_LSTM2_1directional,T3_CNN2_LSTM2_1directional,\
    CNN_LSTM1_2directional,CNN_LSTM2_1directional, hjw_PowerModel,CNN_LSTM1_1directional, hjw_onlyLSTM,New1,\
    CNN_depth1_height1_1000_100_LSTM2,CNN_depth1_height3_1000_100_LSTM2,CNN_depth1_height5_1000_100_LSTM2,\
    CNN_depth2_height3_300_100_LSTM2,CNN_depth2_height1_3_3_5_300_100_LSTM2

import importlib

import os
from torch.optim.lr_scheduler import StepLR


import matplotlib.pyplot as plt
import matplotlib.patches as patches
''' model.train() 和 model.eval()有啥作用 '''
''' net能否.to(device)'''


''' 类别不均衡问题太严重了, 输出都是0,因为样本中就很少的1,导致怎么训练loss都维持在那个值'''
''' 解决方法: ①从损失函数那里可以入手吗, crossentropy有个参数是说这个事情的 ; focal loss
            ②其他方法'''

''' SGD,weight23 train和val loss 都降不下去, 在0.68 
    其他都不变, 只是将SGD换成Adam, 训练loss明显降下去了
    说明优化器首选Adam 
    
    Adam, weight = 1:23, weight_dacay = 0.0001  ->  train_loss = 0.60, min_val_loss = 0.62 ,发现里面1有点多, 可能weight设的有点大了
    Adam, weight = 1:20, weight_dacay = 0.0001  ->  train_loss = 0.591, min_val_loss = 0.65 ,里面1还是有点多
    下一步, 检查model里面的BN和激活函数都设置正确了吗'''



''' loss降不下去,也有可能是data和label并不对应,比如无论我是否shuffle,图像和标签都对不上号 
    我的数据集出现了问题,不封装之前就对不上了 '''

class hjw_home():
    def __init__(self):
        self.batch_size = batch_size
        self.init_lr = 0.01
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        '''loss是一个类,必须先实例化才能使用'''
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.,30.]))
        # self.loss_fn = nn.CrossEntropyLoss()   # <class 'torch.nn.modules.loss.CrossEntropyLoss'>
        self.focalloss_fn = []
        

        self.logging_save_path = r'D:\Neural Network\毕设LSTM\logging'
        self.model_save_path = r'D:\Neural Network\毕设LSTM\Best_model'

    def train(self,epoches):
        '''第一步：先准备数据'''
        train_loader = hjw_PowerDataLoader(self.batch_size,mode = 'train')
        val_loader = hjw_PowerDataLoader(self.batch_size,mode = 'validation')

        '''第二步：定义网络并查看参数量'''
        net = hjw_PowerModel(batch_size)
        net = net.to(self.device)
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in net.parameters())
        print(f"Parameters: {trainable_params} trainable, {total_params} total.")

        '''Step3:定义优化器和学习率更新器'''
        ''' 加了weight_dacay后, trainng_loss反而都降不下去了,现在是0.61'''
        ''' 然而不加weight_dacy, 会导致严重的过拟合, trainning_loss可以到0.34,但是val_loss飘到了2.5'''
        optimizer = optim.Adam(net.parameters(),lr=self.init_lr,weight_decay=0.0003)
        # optimizer = optim.Adam(net.parameters(),lr=self.init_lr)
        # optimizer = optim.SGD(net.parameters(),lr=self.init_lr,momentum=0.9,weight_decay=0.0001)
        scheduler = StepLR(optimizer,25,gamma=0.1)

        log_txt = []
        all_train_loss = []
        all_val_loss = []
        for epoch in range(epoches):
            train_loss = 0 # 1个epoch内所有sample总的loss
            val_loss = 0
            for train_batch_idx,(data,label) in enumerate(train_loader):
                print(f'epoch:{epoch} train_idx:{train_batch_idx}')
                print(type(data),type(label)) # tensor, list
                print(data.size(),len(label)) # torch.Size([18, 90000, 1])   90 其中label的每一个都是一个长为16的tensor
                data = data.view(-1,sub_num,sub_len).float()
                data = data.to(self.device)

                '''优化器清零 '''
                optimizer.zero_grad()

                pred = net(data)
                # print(pred.size())  # torch.Size([18, 140, 2])
                new_pred = torch.permute(pred,(0,2,1))
                
                new_label = []
                for i in range(batch_size):
                    kk = [j[i].item() for j in label]
                    new_label.append(kk)
                
                new_label = torch.tensor(new_label)
                # print(new_label.size()) torch.Size([18, 140])

                
                ''' 这里先不用库本身的nn.CrossEntropyLoss 
                    库本身的nn.CrossEntropyLoss已经平均掉batch_size了,
                    因此最后的loss=0.16就是每个sample在每个二分类问题(共有90个问题)上的loss,
                    也就是85% : 15% 的概率预测正确 '''
                # new_loss = self.loss_fn(new_pred,new_label) 
                # print(new_loss)
                kk = my_CEloss()
                new_loss = kk(new_pred,new_label,30)
                ''' 验证过了, 他的和我的loss值是一样的 '''

                train_loss += new_loss.item() 

                new_loss.backward()
                optimizer.step()
            scheduler.step()
            aver_epoch_train_loss = train_loss/(train_batch_idx+1) # 每个sample的loss


            val_loss = 0
            with torch.no_grad():
                for val_batch_idx,(val_data,val_label) in enumerate(val_loader):
                    print(f'epoch:{epoch} val_idx:{val_batch_idx}')
                    val_data = val_data.view(-1,sub_num,sub_len).float()
                    pred = net(val_data)
                    new_pred = torch.permute(pred,(0,2,1))

                    new_label = []
                    for i in range(batch_size):
                        kk = [j[i].item() for j in val_label]
                        new_label.append(kk)
                    new_label = torch.tensor(new_label)

                    ''' 这里先不用库本身的nn.CrossEntropyLoss '''
                    # new_loss = self.loss_fn(new_pred,new_label)
                    kk = my_CEloss()
                    new_loss = kk(new_pred,new_label,30)

                    val_loss += new_loss.item()

            aver_epoch_val_loss = val_loss / (val_batch_idx+1)

            print(optimizer.state_dict()['param_groups'][0]['lr'])
            print(f'Epoch:{epoch+1}  aver_train_loss = {aver_epoch_train_loss} , aver_val_loss = {aver_epoch_val_loss}')
            log_txt.append(f'Epoch:{epoch+1}  aver_train_loss = {aver_epoch_train_loss} , aver_val_loss = {aver_epoch_val_loss}')
            np.savetxt(os.path.join(self.logging_save_path,'bn_relu_my_CEloss_155_245_weightdecay_0.0003_weight_1_30_loss.txt'),log_txt,fmt = '%s')

            all_train_loss.append(aver_epoch_train_loss)
            all_val_loss.append(aver_epoch_val_loss)

            if aver_epoch_val_loss<= min(all_val_loss):
                torch.save(net.state_dict(),os.path.join(self.model_save_path,'bn_relu_my_CEloss_155_245_weightdecay_0.0003_weight_1_30_model.pt'))
        with open(os.path.join(self.logging_save_path,'bn_relu_my_CEloss_155_245_weightdecay_0.0003_weight_1_30_loss.txt'),mode='a') as f:
            f.write(f'min_val_loss = {min(all_val_loss)}')

    def check(self,path):
        net = hjw_PowerModel(batch_size)
        net.load_state_dict(torch.load(path))
        val_loader = hjw_PowerDataLoader(self.batch_size,mode = 'train')
        with torch.no_grad():
            for val_batch_idx,(val_data,val_label) in enumerate(val_loader):
                print(f'val_idx:{val_batch_idx}')
                # print(val_data.size()) # torch.Size([18, 90000, 1])

                
                
                for i in range(len(val_label)):
                    val_label[i] = val_label[i].numpy()
                label_1 = []


                for i in val_label:
                    label_1.append(i[6])
                print(label_1)
                val_data = val_data.view(-1,sub_num,sub_len).float()
                pred = net(val_data)  # torch.Size([18, 140, 2])
                pred_label_1 = []
                kk = pred[6,:,:]
                
                for i in range(90):
                    if kk[i,0]>kk[i,1]:
                        pred_label_1.append(0)
                    else:
                        pred_label_1.append(1)
                print(pred_label_1)
                print(kk)
                x = np.arange(155000,245000)
                sample = val_data[6,:,0].numpy()
                plt.figure()
                plt.plot(x,sample)
                currentAxis = plt.gca()
                num = begin # 155000
                for kk in label_1:
                    if int(kk) == 1:
                        rect = patches.Rectangle((num,0),1000,0.01,linestyle = 'dotted',edgecolor = 'r',facecolor = 'none')
                    else:
                        rect = patches.Rectangle((num,0),1000,0.005,linestyle = 'dotted',edgecolor = 'g',facecolor = 'none')
                    currentAxis.add_patch(rect)
                    num += 1000
                plt.xlabel('f(Hz)')
                plt.ylabel('Amplitude(V)')
                plt.xticks(range(155000,245000,3000),range(155,245,3)) # 设置坐标刻度，前面是刻度实际宽度，后面是显示出来的文字
                plt.show()
                print(' ')

                plt.figure()
                plt.plot(x,sample)
                currentAxis = plt.gca()
                num = begin
                for kk in pred_label_1:
                    if int(kk) == 1:
                        rect = patches.Rectangle((num,0),1000,0.01,linestyle = 'dotted',edgecolor = 'r',facecolor = 'none')
                    else:
                        rect = patches.Rectangle((num,0),1000,0.005,linestyle = 'dotted',edgecolor = 'g',facecolor = 'none')
                    currentAxis.add_patch(rect)
                    num += 1000
                plt.xlabel('f(Hz)')
                plt.ylabel('Amplitude(V)')
                plt.xticks(range(155000,245000,3000),range(155,245,3))
                plt.show()
                print(' ')


class my_CEloss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    # batch_size = 18
    # input = [18,2,90]
    # label = [18,90]
    def forward(self,input,label,weight):
        result = 0
        for i in range(label.size(0)):
            # mini_batch里面的1个sample
            loss = 0
            pred = input[i,:,:]
            tar = label[i,:].float()
            num = len(tar)
            for j in range(len(tar)):
                if int(tar[j])==0:
                    loss += -torch.log(torch.exp(pred[0,j])/(torch.exp(pred[0,j])+torch.exp(pred[1,j])))
                else:
                    loss += -weight*torch.log(torch.exp(pred[1,j])/(torch.exp(pred[0,j])+torch.exp(pred[1,j])))
                    num += (weight-1)
            loss /= num # 对应nn.CrossEntroptLoss()的reduction = 'mean'
            result += loss # 求和1个mini_batch的总loss
        return result/label.size(0)



import logging
import sys
import math
class hjw_train_agent():
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logging_path = 'D:/Neural Network/毕设LSTM/最优模型试验/logging/'
        self.model_save_path = 'D:/Neural Network/毕设LSTM/最优模型试验/best_model/'
        self.loss_path = 'D:/Neural Network/毕设LSTM/最优模型试验/loss/'
        self.init_lr = 0.01
        self.weight_decay = 0.0003
        self.weight = torch.tensor([1.,2.6,1.])
        self.loss_fn = nn.CrossEntropyLoss(weight=self.weight.to(self.device))

    # 感觉不能训练中用nms，一是开销太大，二是训练最开始，几乎全都是1或0
    ''' 模型训练后,评估的时候用nms 
        input:tensor
                data (sub_num,sub_len) 
                    (90,1000)
                pred (sub_num,class)
                    (90,2) 或 (90,3)
        output:tensor
                NMS_pred (sub_num)
    ''' 
    def hjw_nms(self,data,pred):
        data = data.to('cpu').numpy()  # [90,1000]
        pred = pred.to('cpu').numpy()  # [90,2]
        NMS_pred = []

        pred_i = [np.argmax(pred[k,:]) for k in range(pred.shape[0])]  # list类型
        pred_i = np.array(pred_i)
        out_1 = np.where(pred_i == 1)[0]   # out_1 = [69 75 86]


        out_f = []
        for j in out_1:
            data_j = data[j,:] # data_j = [1000]
            ''' 下面要从子区间中提取中心频率了 '''
            # 1.第1个幅值最大点max_1的位置
            max_1 = np.argmax(data_j)
            # 2.删除max_1左右各100个点内，更新data_jj
            data_jj = data_j.copy()
            data_jj[max(0,max_1-100):min(max_1+100,999)] = 0
            # 3.第2个幅值最大点max_2的位置
            max_2 = np.argmax(data_jj)

            ''' 要是幅值最高点在边缘处,那么他就少了一半幅值较高的点
                因此单纯看数量没有意义,还是要看均值 '''

            mean_1 = np.mean(data_j[max(0,max_1-100):min(max_1+100,999)])
            mean_2 = np.mean(data_j[max(0,max_2-100):min(max_2+100,999)])

            if mean_1>mean_2:
                out_f.append(max_1)
            else:
                out_f.append(max_2)
            
        print(out_1)
        print(out_f)
        # [26 50 51 84]
        # [666, 999, 217, 365]

        out_clustering = []
        done = [-1]
        for pp in range(len(out_1)):
            if not pp<=max(done):
                loc = out_1[pp]
                if loc+1 not in out_1:
                    out_clustering.append({'loc':loc,'f':out_f[pp]})
                    done.append(pp)
                else:

                    start = pp
                    while loc+1  in out_1:
                        end = pp+1
                        loc += 1
                        pp+=1
                    out_clustering.append({'loc':out_1[start:end+1],'f':out_f[start:end+1]})
                    done.append(end)

        print(len(out_clustering))
        print(out_clustering)  # [{'loc': array([ 9, 10, 11], dtype=int64), 'f': [684, 283, 682]}, {'loc': 74, 'f': 16}]

        # for cluster in out_clustering:
        #     loc = cluster['loc']
        #     f = cluster['f']
        #     if len(loc) == 2:


        ''' 对于2个区间连续的, 如果某个f小于100 或 大于 900, 那么肯定可以认为这个区间是旁边区间的附属'''


        print(' ')



        return NMS_pred


    def visualize(self,model_path):
        # net = CNN_LSTM1_1directional(18).to(self.device)
        # net = CNN_LSTM1_2directional(18).to(self.device)
        # net = CNN_LSTM2_1directional(18).to(self.device)
        # net = hjw_PowerModel(18).to(self.device)

        # net = hjw_onlyLSTM(18).to(self.device)
        # net = CNN3_FC().to(self.device)

        # net = CNN_depth2_height3_300_100_LSTM2(18).to(self.device)

        net = CNN_depth2_height1_3_3_5_300_100_LSTM2(18).to(self.device)
        net.load_state_dict(torch.load(model_path))
        # val_loader = hjw_TwoSMPS_DataLoader(batch_size,mode='validation')
        val_loader = hjw_ThreeSMPS_DataLoader(batch_size,mode='validation')

        with torch.no_grad():
            for val_batch_idx,(data,label) in enumerate(val_loader):
                data_ = data.view(-1,sub_num,sub_len).float()
                data_ = data_.to(self.device)

                new_label = []
                for i in range(data_.size(0)):
                    kk = [j[i].item() for j in label]
                    new_label.append(kk)
                new_label = torch.tensor(new_label).to(self.device)

                pred = net(data_)
                ''' 重要形状 
                # data.size = torch.Size([18, 90, 1000])
                # pred.size = torch.Size([18, 90, 3])
                # new_label.size = torch.Size([18, 90])
                '''
                ''' 应用nms,返回numpy [18,90] '''

                for j in range(new_label.size(0)):
                    pred_label_j = [torch.argmax(pred[j,k,:]).item() for k in range(pred.size(1))] # [90]
                    # pred_label_j = pred[j,:]
                    true_label_j = new_label[j,:].to('cpu').numpy()   # [90]
                    data_j = data[j,:].to('cpu').numpy()  # 变换后 = [1,90000]

                    font1 = {'family':'Times New Roman','size':34}
                    x = np.arange(155000,245000)

                    maxv = np.max(data_j)
                    minv = np.min(data_j)
                    data_j = [(pp-minv)/(maxv-minv) for pp in data_j]

                    fig,axes = plt.subplots()
                    plt.plot(x,data_j)
                    # currentAxis = plt.gca()
                    # num = begin # 155000
                    # for kk in true_label_j:
                    #     if int(kk) == 1:
                    #         rect = patches.Rectangle((num,0),1000,0.4,linestyle = 'dotted',edgecolor = 'r',facecolor = 'none')
                    #     elif int(kk) == 2:
                    #         rect = patches.Rectangle((num,0),1000,0.4,linestyle = 'dotted',edgecolor = 'b',facecolor = 'none')
                    #     else:
                    #         rect = patches.Rectangle((num,0),1000,0.1,linestyle = 'dotted',edgecolor = 'g',facecolor = 'none')
                    #     currentAxis.add_patch(rect)
                    #     num += 1000
                    plt.xlabel('f(Hz)',font1)
                    plt.ylabel('Amplitude',font1)
                    # plt.xticks(range(155000,245000,3000),range(155,245,3))
                    plt.tick_params(labelsize = 32)

                    x_kedu = axes.get_xticklabels()
                    [i.set_fontname('Times New Roman') for i in x_kedu]
                    y_kedu = axes.get_yticklabels()
                    [i.set_fontname('Times New Roman') for i in y_kedu]
                    plt.show()
                    print(' ')



                    
                    # fig,axes = plt.subplots()
                    # axes.plot(x,data_j)
                    # currentAxis = plt.gca()
                    # num = begin # 155000
                    # for kk in pred_label_j:
                    #     if int(kk) == 1:
                    #         rect = patches.Rectangle((num,0),1000,0.4,linestyle = 'dotted',edgecolor = 'r',facecolor = 'none')
                    #     elif int(kk) == 2:
                    #         rect = patches.Rectangle((num,0),1000,0.2,linestyle = 'dotted',edgecolor = 'b',facecolor = 'none')
                    #     else:
                    #         rect = patches.Rectangle((num,0),1000,0.1,linestyle = 'dotted',edgecolor = 'g',facecolor = 'none')
                    #     currentAxis.add_patch(rect)
                    #     num += 1000
                    # plt.xlabel('f(Hz)',font1)
                    # plt.ylabel('Amplitude',font1)
                    # # plt.xticks(range(155000,245000,3000),range(155,245,3))
                    # plt.tick_params(labelsize = 18)

                    # x_kedu = axes.get_xticklabels()
                    # [i.set_fontname('Times New Roman') for i in x_kedu]
                    # y_kedu = axes.get_yticklabels()
                    # [i.set_fontname('Times New Roman') for i in y_kedu]
                    # plt.show()
                    # predd = self.hjw_nms(data_[j,:,:],pred[j,:,:])
                    # print(' ')



    def check(self,model_path,log_name):
        
        self.hjw_2logging(filename=self.logging_path+log_name)
        # net = CNN_LSTM1_1directional(18).to(self.device)
        # net = CNN_LSTM1_2directional(18).to(self.device)
        # net = CNN_LSTM2_1directional(18).to(self.device)
        # net = hjw_PowerModel(18).to(self.device)

        # net = hjw_onlyLSTM(18).to(self.device)
        # net = CNN3_FC().to(self.device)

        # net = T1_CNN2_LSTM2_1directional(18).to(self.device)
        # net = T3_CNN2_LSTM2_1directional(18).to(self.device)

        # net = New1(18).to(self.device)
        # net = CNN_depth1_height5_1000_100_LSTM2(18).to(self.device)

        ''' 动态导入Model,大大提高效率 ''' 
        mod = importlib.import_module('hjw_PowerModel')
        net = getattr(mod,model_path[model_path.find('CNN_'):model_path.find('_weight')])
        net = net(18).to(self.device)
        net.load_state_dict(torch.load(model_path))



        logging.info(f'=== Successfully load the Model === : {model_path}')
        # train_loader = hjw_TwoSMPS_DataLoader(batch_size,mode='train')
        # val_loader = hjw_TwoSMPS_DataLoader(batch_size,mode='validation')

        train_loader = hjw_ThreeSMPS_DataLoader(batch_size,mode = 'train')
        val_loader = hjw_ThreeSMPS_DataLoader(batch_size,mode = 'validation')
        logging.info('=== Successfully load the DataLoader ===')

        train_acc = []
        train_precision = []
        train_recall = []
        with torch.no_grad():
            for train_batch_idx,(data,label) in enumerate(train_loader):
                data = data.view(-1,sub_num,sub_len).float()
                # print(f'data.size = {data.size()}')
                data = data.to(self.device)

                new_label = []
                for i in range(data.size(0)):
                    kk = [j[i].item() for j in label]
                    new_label.append(kk)
                new_label = torch.tensor(new_label).to(self.device)

                pred = net(data)
                ''' 重要形状 
                # data.size = torch.Size([18, 90, 1000])
                # pred.size = torch.Size([18, 90, 3])
                # new_label.size = torch.Size([18, 90])
                '''

                # new_pred = torch.permute(pred,(0,2,1)).to(self.device)
                
                batch_acc = []
                batch_precision = []
                batch_recall = []
                for j in range(new_label.size(0)):

                    sample_yes = 0
                    sample_TP = 0
                    sample_TPFP = 0
                    sample_TPFN = 0
                    sample_total = 90
                    pred_label_j = [torch.argmax(pred[j,k,:]).item() for k in range(pred.size(1))]
                    true_label_j = new_label[j,:].to('cpu').numpy()


                    for k in range(90):
                        if pred_label_j[k] == true_label_j[k]:
                            sample_yes += 1
                        if pred_label_j[k]==1 :
                            sample_TPFP += 1
                        if pred_label_j[k]==1 and true_label_j[k]==1:
                            sample_TP += 1
                        if true_label_j[k]==1:
                            sample_TPFN += 1
                    
                    sample_acc = sample_yes/sample_total
                    if sample_TPFP == 0:
                        sample_precision = 0 
                    else:
                        sample_precision = sample_TP/sample_TPFP
                    
                    if sample_TPFN == 0:
                        if sample_TP == 0 :
                            sample_recall = 1
                        else:
                            sample_recall = 0
                    else:
                        sample_recall = sample_TP/sample_TPFN

                    # print(sample_acc,sample_precision,sample_recall)
                    batch_acc.append(sample_acc)
                    batch_precision.append(sample_precision)
                    batch_recall.append(sample_recall)
                
                ''' 这三个是每个batch的平均数据 '''
                this_batch_acc = np.mean(batch_acc)
                this_batch_precision = np.mean(batch_precision)
                this_batch_recall = np.mean(batch_recall)

                train_acc.append(this_batch_acc)
                train_precision.append(this_batch_precision)
                train_recall.append(this_batch_recall)
        
        last_train_acc = np.mean(train_acc)
        last_train_precision = np.mean(train_precision)
        last_train_recall = np.mean(train_recall)

        # print(last_train_acc,last_train_precision,last_train_recall)


        val_acc = []
        val_precision = []
        val_recall = []
        with torch.no_grad():
            for val_batch_idx,(data,label) in enumerate(val_loader):
                data = data.view(-1,sub_num,sub_len).float()
                # print(f'data.size = {data.size()}')
                data = data.to(self.device)

                new_label = []
                for i in range(data.size(0)):
                    kk = [j[i].item() for j in label]
                    new_label.append(kk)
                new_label = torch.tensor(new_label).to(self.device)

                pred = net(data)
                ''' 重要形状 
                # data.size = torch.Size([18, 90, 1000])
                # pred.size = torch.Size([18, 90, 3])
                # new_label.size = torch.Size([18, 90])
                '''

                # new_pred = torch.permute(pred,(0,2,1)).to(self.device)
                
                batch_acc = []
                batch_precision = []
                batch_recall = []
                for j in range(new_label.size(0)):

                    sample_yes = 0
                    sample_TP = 0
                    sample_TPFP = 0
                    sample_TPFN = 0
                    sample_total = 90
                    pred_label_j = [torch.argmax(pred[j,k,:]).item() for k in range(pred.size(1))]
                    true_label_j = new_label[j,:].to('cpu').numpy()

                    # print(pred_label_j)
                    # print(true_label_j)

                    for k in range(90):
                        if pred_label_j[k] == true_label_j[k]:
                            sample_yes += 1
                        if pred_label_j[k]==1 :
                            sample_TPFP += 1
                        if pred_label_j[k]==1 and true_label_j[k]==1:
                            sample_TP += 1
                        if true_label_j[k]==1:
                            sample_TPFN += 1

                    sample_acc = sample_yes/sample_total
                    if sample_TPFP == 0:
                        sample_precision = 0 
                    else:
                        sample_precision = sample_TP/sample_TPFP
                    # sample_recall = sample_TP/sample_TPFN
                    if sample_TPFN == 0:
                        if sample_TP == 0 :
                            sample_recall = 1
                        else:
                            sample_recall = 0
                    else:
                        sample_recall = sample_TP/sample_TPFN

                    batch_acc.append(sample_acc)
                    batch_precision.append(sample_precision)
                    batch_recall.append(sample_recall)
                
                ''' 这三个是每个batch的平均数据 '''
                this_batch_acc = np.mean(batch_acc)
                this_batch_precision = np.mean(batch_precision)
                this_batch_recall = np.mean(batch_recall)

                val_acc.append(this_batch_acc)
                val_precision.append(this_batch_precision)
                val_recall.append(this_batch_recall)
        
        last_val_acc = np.mean(val_acc)
        last_val_precision = np.mean(val_precision)
        last_val_recall = np.mean(val_recall)
        # print(last_val_acc)
        # print(last_val_acc,last_val_precision,last_val_recall)
        logging.info(f'train: acc = {last_train_acc}, precision = {last_train_precision}, recall = {last_train_recall}')
        logging.info(f'val: acc = {last_val_acc}, precision = {last_val_precision}, recall = {last_val_recall}')

    def train(self,log_filename,\
                    epoches=160,\
                    batch_size=18,\
                    label_path=r'D:\毕业设计\数据采集平台\label.txt',\
                    train_data_path=r'D:\毕业设计\数据采集平台\new_train_data\data',\
                    val_data_path=r'D:\毕业设计\数据采集平台\new_val_data\data'):
        
        self.hjw_2logging(self.logging_path + log_filename)

        # train_loader = hjw_TwoSMPS_DataLoader(batch_size,mode='train')
        # val_loader = hjw_TwoSMPS_DataLoader(batch_size,mode='validation')

        logging.info(f'weigth = {self.weight}')
        train_loader = hjw_ThreeSMPS_DataLoader(batch_size,mode = 'train')
        val_loader = hjw_ThreeSMPS_DataLoader(batch_size,mode = 'validation')
        logging.info('=== Successfully load the dataset ===')

        mod = importlib.import_module('hjw_PowerModel')
        net = getattr(mod,log_filename[0:log_filename.find('_weight')])
        net = net(batch_size).to(self.device)
        total_num = sum(param.numel() for param in net.parameters())
        trainabel_num = sum(param.numel() for param in net.parameters() if param.requires_grad)
        logging.info(f"=== Successfully load the Model:  {log_filename[0:log_filename.find('_weight')]}===")
        logging.info(f'Parameters: {total_num} total, {trainabel_num} trainable')

        optimizer = optim.Adam(net.parameters(),lr = self.init_lr,weight_decay=self.weight_decay)
        logging.info(f'Using optimizer: {optimizer}')

        scheduler = StepLR(optimizer,25,gamma=0.1)
        logging.info(f'Using scheduler: {scheduler}')


        all_train_loss = []
        all_val_loss = []
        for epoch in range(epoches):
            train_loss = 0
            val_loss = 0
            for train_batch_idx,(data,label) in enumerate(train_loader):
                # print(data.size())  # torch.Size([16, 90000, 1])
                print(f'train_idx = {train_batch_idx}')
                # print(len(label))  # 90,每一个都是长为16的tensor

                # data = data.view(-1,sub_len,sub_num).float()
                data = data.view(-1,sub_num,sub_len).float().to(self.device)


                new_label = []
                for i in range(data.size(0)):
                    kk = [j[i].item() for j in label]
                    new_label.append(kk)
                new_label = torch.tensor(new_label).to(self.device)

                optimizer.zero_grad()

                pred = net(data)
                ''' 重要形状 
                # data.size = torch.Size([18, 90, 1000])
                # pred.size = torch.Size([18, 90, 2])
                # new_label.size = torch.Size([18, 90])
                '''

                new_pred = torch.permute(pred,(0,2,1)).to(self.device)
                # print(pred.size())  # torch.Size([16, 90, 2])

                # kk = my_CEloss()
                # loss = kk(new_pred,new_label,30)
                # print(loss1)
                loss = self.loss_fn(new_pred,new_label) 
                # print(loss)

                train_loss += loss.item()

                loss.backward()
                optimizer.step()
            
            scheduler.step()
            aver_epoch_train_loss = train_loss/(train_batch_idx+1) # 平均每个sample在每个二分类问题上的loss

            with torch.no_grad():
                for val_batch_idx,(data,label) in enumerate(val_loader):
                    print(f'val_idx = {val_batch_idx}')
                    # data = data.view(-1,sub_len,sub_num).float()
                    data = data.view(-1,sub_num,sub_len).float()
                    data = data.to(self.device)
                    pred = net(data)
                    new_pred = torch.permute(pred,(0,2,1)).to(self.device)
                    new_label = []
                    for i in range(data.size(0)):
                        kk = [j[i].item() for j in label]
                        new_label.append(kk)
                
                    new_label = torch.tensor(new_label).to(self.device)
                    # kk = my_CEloss()
                    # new_loss = kk(new_pred,new_label,30)
                    new_loss = self.loss_fn(new_pred,new_label) 

                    val_loss += new_loss.item()
            
            aver_epoch_val_loss = val_loss / (val_batch_idx+1)

            logging.info(f'Epoch:{epoch+1}  aver_train_loss = {aver_epoch_train_loss} , aver_val_loss = {aver_epoch_val_loss}')
            logging.info(f" lr = {optimizer.state_dict()['param_groups'][0]['lr']}")

            all_train_loss.append(aver_epoch_train_loss)
            all_val_loss.append(aver_epoch_val_loss)

            if aver_epoch_val_loss<= min(all_val_loss):
                torch.save(net.state_dict(),os.path.join(self.model_save_path,log_filename[0:-4]+'_min_val.pt'))
            if aver_epoch_train_loss<= min(all_train_loss):
                torch.save(net.state_dict(),os.path.join(self.model_save_path,log_filename[0:-4]+'_min_train.pt'))
            

            min_val_loc = np.argmin(all_val_loss)
            later = len(all_val_loss)-min_val_loc
            if later >=90:
                break
        logging.info(f'=== Trained {epoch+1} epoches ===')
        logging.info(f'min_train_loss = {min(all_train_loss)}, min_val_loss = {min(all_val_loss)}')
        np.save(self.loss_path + log_filename[0:-4] + '_all_train_loss.npy',all_train_loss)
        np.save(self.loss_path + log_filename[0:-4] + '_all_val_loss.npy',all_val_loss)

    def hjw_2logging(self,filename,file_level=logging.INFO,console_level=logging.INFO): # 注意缺省值INFO的大小写
        file_handler = logging.FileHandler(filename,mode = 'a')
        file_handler.setFormatter(logging.Formatter('[%(asctime)s %(levelname)s] %(message)s',datefmt='%m/%d/%Y %H:%M:%S'))
        file_handler.setLevel(file_level)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('[%(asctime)s %(levelname)s] %(message)s',datefmt='%m/%d/%Y %H:%M:%S'))
        console_handler.setLevel(console_level)

        logging.basicConfig(level=min(file_level,console_level),handlers=[file_handler,console_handler])


if __name__ == '__main__':


    ta = hjw_train_agent()
    ''' 可视化命令 '''
    # ta.visualize(model_path=r'D:\Neural Network\毕设LSTM\最优模型试验\best_model\CNN_depth2_height1_3_3_5_300_100_LSTM2_weight4_1++++_min_val.pt')
    
    
    ''' train之前, 注意修改Model里面全连接层输出2分类,还是3分类 
                    self.loss_fn的权重也要修改,是2个还是3个 '''
    # 在双电源的2LSTM_1direction 的最好值：train_loss = 0.004,   val_loss = 0.012
    # 三电源不增强：      0.00412 0.1686
    # 三电源addtestdata： 0.00411 0.1656
    # 三电源的scale0.05增强： 0.00124 0.1683
    # 三电源的scale0.1增强：  0.00125 0.1963
    # 三电源的scale0.05+randomnoise增强： 0.00087  0.1358
    # 三电源的randomnoise增强：0.0019 0.1740
    # 三电源的slicing增强：   0.00267 0.1718
    # 三电源的slicing+randomnoise增强：   0.00191  0.1900
    

    ''' 千万注意: 前面self.fn的weight也要改 '''
    ta.train(log_filename='CNN_depth2_height1_1_300_100_LSTM2_weight2_x6_1++++.txt')

    ''' 获取准确率、精确率、召回率 命令 '''
    ta.check(model_path='D:/Neural Network/毕设LSTM/最优模型试验/best_model/CNN_depth2_height1_1_300_100_LSTM2_weight2_x6_1++++_min_val.pt',\
        log_name='CNN_depth2_height1_1_300_100_LSTM2_weight2_x6_1++++.txt')



    
    # a = hjw_home()
    # a.train(150) 
    # a.check(r'D:\Neural Network\毕设LSTM\Best_model\bn_relu_correct_her_CEloss_cnn_155_245_weight_0.0003_1_30_model.pt')


    # import matplotlib.pyplot as plt
    # a = r'D:\Neural Network\毕设LSTM\logging\cnn_155_245_weight_0.0003_1_16.5_loss.txt'
    # b = np.loadtxt(a,dtype=str)
    # x = np.arange(1,151)
    # train_loss = [float(i[3]) for i in b]
    # val_loss = [float(i[7]) for i in b]
    # plt.plot(x,train_loss,label = 'Training_loss')
    # plt.plot(x,val_loss,label='Val_loss')
    # plt.xlabel('Epoch',fontsize = 14)
    # plt.legend()
    # plt.show()


    '''验证过了,最基本的2个数,CEloss和我想的是一样的'''
    # a = torch.tensor([[0.1,0.9]])
    # a = a.unsqueeze(2)
    # target = torch.tensor([0]).unsqueeze(0)
    # fn1 = nn.CrossEntropyLoss(reduction='sum')
    # fn2 = nn.CrossEntropyLoss(weight=torch.tensor([1.,5.]),reduction='sum')
    
    # out1 = fn1(a,target)
    # out2 = fn2(a,target)
    # print(out1,out2)
    # me = -5*log(exp(0.1)/(exp(0.9)+exp(0.1)))
    # print(me)

    # # tensor(1.1711) tensor(1.1711)
    # # 5.855503329738889
    








