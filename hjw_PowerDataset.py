


'''划分训练集,验证集,测试集:  7:2:1   182 = 128 + 36 + 18'''
# data_path = r'D:\毕业设计\数据采集平台\Frequency_Sequences'
# label_path = r'D:\毕业设计\数据采集平台\label.txt'

# num = list(range(1,183))  # 最后到182
# train = random.sample(num,128)
# num_ = [i for i in num if not i in train]
# vali = random.sample(num_,36)
# test = [i for i in num_ if not i in vali]

# train_path = r'D:\毕业设计\数据采集平台\new_train_data'
# vali_path = r'D:\毕业设计\数据采集平台\new_val_data'
# test_path = r'D:\毕业设计\数据采集平台\new_test_data'


# with open(label_path) as f:
#     label = f.readlines() # 这样读出来的label是list格式的
# label_train = []
# label_vali = []
# label_test = []
# for i in os.listdir(data_path):
#     kk = int(i[0:i.find('.')])
#     if kk in train:
#         shutil.copy( os.path.join(data_path,i) , os.path.join(train_path,'data') )
#         label_train.append(label[kk-1].strip())
#     elif kk in vali:
#         shutil.copy( os.path.join(data_path,i) , os.path.join(vali_path,'data') )
#         label_vali.append(label[kk-1].strip())
#     else:
#         shutil.copy( os.path.join(data_path,i) , os.path.join(test_path,'data') )
#         label_test.append(label[kk-1].strip())

# np.savetxt(os.path.join(train_path,'label_train.txt'),label_train,fmt ='%s')
# np.savetxt(os.path.join(vali_path,'label_val.txt'),label_vali,fmt = '%s')
# np.savetxt(os.path.join(test_path,'label_test.txt'),label_test,fmt = '%s')

# print(' ')
'''检查各个数据集中是否有重复的'''
# path1 = r'D:\毕业设计\数据采集平台\new_train_data\data'
# path2 = r'D:\毕业设计\数据采集平台\new_val_data\data'
# path3 = r'D:\毕业设计\数据采集平台\new_test_data\data'
# a = os.listdir(path1)
# b = os.listdir(path2)
# c = os.listdir(path3)
# for i in a:
#     if i in b or i in c:
#         print('NO')
# for i in b:
#     if i in a or i in c:
#         print('NO')
# for i in c:
#     if i in a or i in b:
#         print('NO')

import math
import numpy as np
from torch.utils.data import Dataset,DataLoader
import os
import scipy.io
import matplotlib.pyplot as plt
from hjw_Settings import begin,end,sub_len,sub_num,batch_size

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def hjw_TwoSMPS_DataLoader(batch_size:int,mode:str):
    dataset = hjw_TwoSMPS_Set(mode)
    if mode == 'train':
        Loader = DataLoader(dataset,batch_size,shuffle=True)
    else:
        Loader = DataLoader(dataset,batch_size,shuffle=False)
    return Loader


def hjw_ThreeSMPS_DataLoader(batch_size:int,mode:str):
    dataset = hjw_ThreeSMPS_Set(mode)
    Loader = DataLoader(dataset,batch_size,shuffle=True)
    return Loader

class hjw_TwoSMPS_Set(Dataset):
    def __init__(self,mode) -> None:
        super().__init__()
        self.label_path = r'D:\毕业设计\数据采集平台\谢哥的数据\labels_fre'
        if mode == 'train':
            self.data_path = r'D:\毕业设计\数据采集平台\谢哥的数据\train'
        elif mode == 'validation':
            self.data_path = r'D:\毕业设计\数据采集平台\谢哥的数据\val'
        elif mode == 'test':
            self.data_path = r'D:\毕业设计\数据采集平台\谢哥的数据\test'
        else:
            raise Exception('Please give a right mode.')

        # 原始数据截取的总的频域区间
        self.begin = 150000
        self.end = 300000
        # 每个小区间长度为1000
        self.sub_len = sub_len

        self.data_dir = []
        self.label_dir = []


        for i in range(len(os.listdir(self.data_path))):
            print(i)
            full_path = os.path.join(self.data_path,os.listdir(self.data_path)[i])

            # 原始频域区间：150000 - 300000
            # 截取频域区间：155000 - 245000 ，长度90000       
            data = np.loadtxt(full_path,dtype=float,delimiter=',')[5000:95000]                    
            self.data_dir.append(data)
            # x = np.arange(155000,245000)
            # plt.plot(x,data)
            # plt.show()
            # print(data.shape) # (90000,)

            name = os.listdir(self.data_path)[i][0:-4]
            # print(name)  # 11-25-17-25-59_p31
            with open(os.path.join(self.label_path,name+'.txt'),'r') as f:
                label = f.read().splitlines()
            
            # print(label) # ['68768,69708,150000,p3', '85705,86038,150000,p1']

            binary_label = [0]*90
            if int(label[0][label[0].find(',')+1:-10]) - int(label[0][0:label[0].find(',')]) >= 2000:
                raise Exception()
            if int(label[1][label[1].find(',')+1:-10]) - int(label[1][0:label[1].find(',')]) >= 2000:
                raise Exception()
            a = int(label[0][0:label[0].find(',')])/1000
            b = int(label[0][label[0].find(',')+1:-10])/1000
            c = int(label[1][0:label[1].find(',')])/1000
            d = int(label[1][label[1].find(',')+1:-10])/1000


            binary_label[math.floor(a)-5] = 2
            binary_label[math.floor(b)-5] = 2
            binary_label[math.floor(c)-5] = 1
            binary_label[math.floor(d)-5] = 1
            ''' 基于NMS的产生二进制label的方法: 取中间点, 感觉不太好 '''
            # loc1 = (int(label[0][0:label[0].find(',')])+int(label[0][label[0].find(',')+1:-10]))/2 + self.begin
            # loc2 = (int(label[1][0:label[1].find(',')])+int(label[1][label[1].find(',')+1:-10]))/2 + self.begin
            # loc_label = [loc1,loc2]
            # # print(loc_label)
            
            # binary_label = [0]*90

            # kk =  (loc1 - self.begin)/self.sub_len
            # xiaoshu, zhengshu = math.modf(kk)

            # zhengshu = int(zhengshu)
            # binary_label[zhengshu] = 2
            # if xiaoshu >0.7:
            #     binary_label[zhengshu+1] = 2
            # elif xiaoshu < 0.3:
            #     binary_label[zhengshu-1] = 2


            # kk =  (loc2 - self.begin)/self.sub_len
            # xiaoshu, zhengshu = math.modf(kk)

            # zhengshu = int(zhengshu)
            # binary_label[zhengshu] = 1
            # if xiaoshu >0.9:
            #     binary_label[zhengshu+1] = 1
            # elif xiaoshu < 0.1:
            #     binary_label[zhengshu-1] = 1


            ''' visualize ''' 
            # font1 = {'family':'Times New Roman','size':22}
            # x = np.arange(155000,245000)
            # plt.plot(x,data)
            # plt.show()
            
            # data_j = data
            # true_label_j = binary_label
            # fig,axes = plt.subplots()
            # axes.plot(x,data_j)
            # currentAxis = plt.gca()
            # num = begin # 155000
            # for kk in true_label_j:
            #     if int(kk) == 2:
            #         rect = patches.Rectangle((num,0),1000,0.02,linestyle = 'dotted',edgecolor = 'r',facecolor = 'none')
            #     elif int(kk) == 1:
            #         rect = patches.Rectangle((num,0),1000,0.02,linestyle = 'dotted',edgecolor = 'b',facecolor = 'none')
            #     else:
            #         rect = patches.Rectangle((num,0),1000,0.01,linestyle = 'dotted',edgecolor = 'g',facecolor = 'none')
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
            ''''''


            # print(binary_label)
            self.label_dir.append(binary_label)

    def __len__(self):
        return len(self.data_dir)
    
    def __getitem__(self,index):
        return self.data_dir[index],self.label_dir[index]




'''原始数据集频域区间:130000 -> 270000
我划分为140个子区间,区间长度为1000'''
class hjw_ThreeSMPS_Set(Dataset):
    def __init__(self,mode) -> None:
        super().__init__()
        # self.label_path = r'D:\毕业设计\数据采集平台\label.txt'
        self.label_path = r'D:\毕业设计\数据采集平台\label_修正后的完整label.txt'
        if mode == 'train':
            self.data_path = r'D:\毕业设计\数据采集平台\new_train_data++\data'
        elif mode == 'validation':
            self.data_path = r'D:\毕业设计\数据采集平台\new_val_data++\data'
        elif mode == 'test':
            self.data_path = r'D:\毕业设计\数据采集平台\new_test_data\data'
        else:
            raise Exception('Please give a right mode.')

        # 原始数据截取的总的频域区间
        self.begin = begin # 155000
        self.end = end     # 245000
        # 每个小区间长度为1000
        self.sub_len = sub_len

        self.data_dir = []
        self.label_dir = []

        with open(self.label_path,'r') as f:
            label = f.read().splitlines()
        
        total_num = len(os.listdir(self.data_path))
        doing = 0
        for path_i in os.listdir(self.data_path):
            full_path = os.path.join(self.data_path,path_i)
            # print(full_path)
            # D:\毕业设计\数据采集平台\new_train_data++\data\101.mat
            # D:\毕业设计\数据采集平台\new_train_data++\data\101_used_scale.mat

            if path_i.find('used') == -1:
                num = int(path_i[0:-4])
                frequency_data = scipy.io.loadmat(full_path)['b']  # 频域区间130000 - 270000 ，长度140000        
            
                frequency_data = frequency_data[25000:-25000]  # 频域区间155000 - 245000 ，长度90000
                # maxv = np.max(frequency_data)
                # minv = np.min(frequency_data)
                # frequency_data = np.array([(pp-maxv)/(maxv-minv) for pp in frequency_data])
                self.data_dir.append(frequency_data)
                data_label = label[num-1].split(',')
                # print(data_label)  # ['181.5:182.5', '231.5:232.5']
                binary_label = [0]*90

                for label_j in data_label:
                    fa = math.floor(float(label_j[0:label_j.find(':')]))
                    fb = math.ceil(float(label_j[label_j.find(':')+1:]))

                    fa -= 155
                    fb -= 155
                    # print(fa,fb)
                    for pp in range(fa,fb):
                        binary_label[pp] = 1
                



                ''' 第一个位置的噪声 '''
                x163_164 = [93,94]
                x163_165 = np.hstack((np.arange(68,93),np.array([41,95,96,97,98])))
                x164_166 = [3,5,6,7,8,9,10]
                x165_166 = [2,4]
                # 最多的 164-165
                if num in x163_164:
                    if binary_label[163-155] != 1:
                        binary_label[163-155] =2
                elif num in x163_165:
                    if binary_label[163-155] != 1:
                        binary_label[163-155] =2
                    if binary_label[164-155] != 1:
                        binary_label[164-155] =2
                elif num in x164_166:
                    if binary_label[164-155] != 1:
                        binary_label[164-155] =2
                    if binary_label[165-155] != 1:
                        binary_label[165-155] =2
                elif num in x165_166:
                    if binary_label[165-155] != 1:
                        binary_label[165-155] =2
                else:
                    if binary_label[164-155] != 1:
                        binary_label[164-155] =2

                ''' 第二个位置的噪声167-168,没有争议,如果那个位置没有电源,就认为是噪声'''
                # if binary_label[167-155] != 1:
                #     binary_label[167-155] = 2
                

                ''' 第三个位置的噪声 , 大多数是228-229 '''
                x229_230 = [2,3,4,5,6,7,8,9,10]
                if num in x229_230:
                    if binary_label[229-155]!=1:
                        binary_label[229-155]=2
                else:
                    if binary_label[228-155]!=1:
                        binary_label[228-155]=2

                ''' 最后一个位置的噪声230-231,,没有争议,如果那个位置没有电源,就认为是噪声'''
                # if binary_label[230-155] != 1:
                #     binary_label[230-155] = 2

                ''' 验证label是否正确 '''
                # if path_i == '15.mat':
                #     maxv = np.max(frequency_data)
                #     minv = np.min(frequency_data)
                #     frequency_data = np.array([(pp-minv)/(maxv-minv) for pp in frequency_data])
                #     x = np.arange(155000,245000)
                #     fig,axes = plt.subplots()
                #     plt.plot(x,frequency_data)
                #     # currentAxis = plt.gca()
                #     # num = 155000
                #     # for kk in range(90):
                #     #     if binary_label[kk] == 1:
                #     #         rect = patches.Rectangle((num,0),1000,0.4,linestyle = 'dotted',edgecolor = 'b',facecolor = 'none',lw = 1.4)
                #     #     elif binary_label[kk] == 2:
                #     #         rect = patches.Rectangle((num,0),1000,0.2,linestyle = 'dotted',edgecolor = 'g',facecolor = 'none',lw=1.4)
                #     #     else:
                #     #         rect = patches.Rectangle((num,0),1000,0.2,linestyle = 'dotted',edgecolor = 'g',facecolor = 'none',lw=1.4)
                #     #     currentAxis.add_patch(rect)
                #     #     num += 1000
                #     # plt.title(path_i)
                #     font1 = {'family':'Times New Roman','size':36}
                #     plt.xlabel('f(Hz)',font1)
                #     plt.ylabel('Amplitude',font1)
                #     plt.tick_params(labelsize = 32)

                #     # plt.xticks(np.arange(155000,246000,2000),np.arange(155,246,2))
                #     x_kedu = axes.get_xticklabels()
                #     [i.set_fontname('Times New Roman') for i in x_kedu]
                #     y_kedu = axes.get_yticklabels()
                #     [i.set_fontname('Times New Roman') for i in y_kedu]
                #     plt.show()
                #     print(' ')
                self.label_dir.append(binary_label)
                doing += 1
                print(f'{doing}/{total_num}')



                continue


            if path_i.find('used_fakenoise1') != -1:
                continue
                num = int( path_i[0:path_i.find('_used_fakenoise1')])
                frequency_data = scipy.io.loadmat(full_path)['b']   # fakenoise1后的：频域区间155000 - 245000 ，长度90000
                # maxv = np.max(frequency_data)
                # minv = np.min(frequency_data)
                # frequency_data = np.array([(pp-maxv)/(maxv-minv) for pp in frequency_data])
                self.data_dir.append(frequency_data)

                data_label = label[num-1].split(',')
                # print(data_label)  # ['181.5:182.5', '231.5:232.5']

                binary_label = [0]*90
                for label_j in data_label:
                    fa = math.floor(float(label_j[0:label_j.find(':')]))
                    fb = math.ceil(float(label_j[label_j.find(':')+1:]))

                    fa -= 155
                    fb -= 155
                    # print(fa,fb)
                    for pp in range(fa,fb):
                        binary_label[pp] = 1
                
                self.label_dir.append(binary_label)
                doing += 1
                print(f'{doing}/{total_num}')
                continue

            if path_i.find('used_fakenoise2') != -1:
                continue
                num = int( path_i[0:path_i.find('_used_fakenoise2')])
                frequency_data = scipy.io.loadmat(full_path)['b']   # fakenoise1后的：频域区间155000 - 245000 ，长度90000
                # maxv = np.max(frequency_data)
                # minv = np.min(frequency_data)
                # frequency_data = np.array([(pp-maxv)/(maxv-minv) for pp in frequency_data])
                self.data_dir.append(frequency_data)

                data_label = label[num-1].split(',')
                # print(data_label)  # ['181.5:182.5', '231.5:232.5']

                binary_label = [0]*90
                for label_j in data_label:
                    fa = math.floor(float(label_j[0:label_j.find(':')]))
                    fb = math.ceil(float(label_j[label_j.find(':')+1:]))

                    fa -= 155
                    fb -= 155
                    # print(fa,fb)
                    for pp in range(fa,fb):
                        binary_label[pp] = 1
                
                self.label_dir.append(binary_label)
                doing += 1
                print(f'{doing}/{total_num}')
                continue


            if path_i.find('used_scale') != -1:
                # continue
                num = int( path_i[0:path_i.find('_used_scale')])
                frequency_data = scipy.io.loadmat(full_path)['b']   # fakenoise1后的：频域区间155000 - 245000 ，长度90000
                # maxv = np.max(frequency_data)
                # minv = np.min(frequency_data)
                # frequency_data = np.array([(pp-maxv)/(maxv-minv) for pp in frequency_data])
                self.data_dir.append(frequency_data)

                data_label = label[num-1].split(',')
                # print(data_label)  # ['181.5:182.5', '231.5:232.5']

                binary_label = [0]*90
                for label_j in data_label:
                    fa = math.floor(float(label_j[0:label_j.find(':')]))
                    fb = math.ceil(float(label_j[label_j.find(':')+1:]))

                    fa -= 155
                    fb -= 155
                    # print(fa,fb)
                    for pp in range(fa,fb):
                        binary_label[pp] = 1
                
                ''' 第一个位置的噪声 '''
                x163_164 = [93,94]
                x163_165 = np.hstack((np.arange(68,93),np.array([41,95,96,97,98])))
                x164_166 = [3,5,6,7,8,9,10]
                x165_166 = [2,4]
                # 最多的 164-165
                if num in x163_164:
                    if binary_label[163-155] != 1:
                        binary_label[163-155] =2
                elif num in x163_165:
                    if binary_label[163-155] != 1:
                        binary_label[163-155] =2
                    if binary_label[164-155] != 1:
                        binary_label[164-155] =2
                elif num in x164_166:
                    if binary_label[164-155] != 1:
                        binary_label[164-155] =2
                    if binary_label[165-155] != 1:
                        binary_label[165-155] =2
                elif num in x165_166:
                    if binary_label[165-155] != 1:
                        binary_label[165-155] =2
                else:
                    if binary_label[164-155] != 1:
                        binary_label[164-155] =2

                ''' 第二个位置的噪声167-168,没有争议,如果那个位置没有电源,就认为是噪声'''
                # if binary_label[167-155] != 1:
                #     binary_label[167-155] = 2
                

                ''' 第三个位置的噪声 , 大多数是228-229 '''
                x229_230 = [2,3,4,5,6,7,8,9,10]
                if num in x229_230:
                    if binary_label[229-155]!=1:
                        binary_label[229-155]=2
                else:
                    if binary_label[228-155]!=1:
                        binary_label[228-155]=2

                ''' 最后一个位置的噪声230-231,,没有争议,如果那个位置没有电源,就认为是噪声'''
                # if binary_label[230-155] != 1:
                #     binary_label[230-155] = 2

                self.label_dir.append(binary_label)
                doing += 1
                print(f'{doing}/{total_num}')
                continue

            if path_i.find('used_randomnoise') != -1:
                # continue
                num = int( path_i[0:path_i.find('_used_randomnoise')])
                frequency_data = scipy.io.loadmat(full_path)['b']   # fakenoise1后的：频域区间155000 - 245000 ，长度90000
                # maxv = np.max(frequency_data)
                # minv = np.min(frequency_data)
                # frequency_data = np.array([(pp-maxv)/(maxv-minv) for pp in frequency_data])
                self.data_dir.append(frequency_data)

                data_label = label[num-1].split(',')
                # print(data_label)  # ['181.5:182.5', '231.5:232.5']

                binary_label = [0]*90
                for label_j in data_label:
                    fa = math.floor(float(label_j[0:label_j.find(':')]))
                    fb = math.ceil(float(label_j[label_j.find(':')+1:]))

                    fa -= 155
                    fb -= 155
                    # print(fa,fb)
                    for pp in range(fa,fb):
                        binary_label[pp] = 1
                
                ''' 第一个位置的噪声 '''
                x163_164 = [93,94]
                x163_165 = np.hstack((np.arange(68,93),np.array([41,95,96,97,98])))
                x164_166 = [3,5,6,7,8,9,10]
                x165_166 = [2,4]
                # 最多的 164-165
                if num in x163_164:
                    if binary_label[163-155] != 1:
                        binary_label[163-155] =2
                elif num in x163_165:
                    if binary_label[163-155] != 1:
                        binary_label[163-155] =2
                    if binary_label[164-155] != 1:
                        binary_label[164-155] =2
                elif num in x164_166:
                    if binary_label[164-155] != 1:
                        binary_label[164-155] =2
                    if binary_label[165-155] != 1:
                        binary_label[165-155] =2
                elif num in x165_166:
                    if binary_label[165-155] != 1:
                        binary_label[165-155] =2
                else:
                    if binary_label[164-155] != 1:
                        binary_label[164-155] =2

                ''' 第二个位置的噪声167-168,没有争议,如果那个位置没有电源,就认为是噪声'''
                # if binary_label[167-155] != 1:
                #     binary_label[167-155] = 2
                

                ''' 第三个位置的噪声 , 大多数是228-229 '''
                x229_230 = [2,3,4,5,6,7,8,9,10]
                if num in x229_230:
                    if binary_label[229-155]!=1:
                        binary_label[229-155]=2
                else:
                    if binary_label[228-155]!=1:
                        binary_label[228-155]=2

                ''' 最后一个位置的噪声230-231,,没有争议,如果那个位置没有电源,就认为是噪声'''
                # if binary_label[230-155] != 1:
                #     binary_label[230-155] = 2

                self.label_dir.append(binary_label)
                doing += 1
                print(f'{doing}/{total_num}')
                continue


            if path_i.find('used_slicing') != -1:
                continue
                num = int( path_i[0:path_i.find('_used_slicing')])
                frequency_data = scipy.io.loadmat(full_path)['b']   # 频域区间155000 - 245000 ，长度90000
                self.data_dir.append(frequency_data)
                # x = np.arange(155000,245000)
                # plt.plot(x,frequency_data)
                # plt.show()


                data_label = label[num-1].split(',')
                # print(data_label)  # ['181.5:182.5', '231.5:232.5']
                binary_label = [0]*90

                for label_j in data_label:
                    fa = math.floor(float(label_j[0:label_j.find(':')]))
                    fb = math.ceil(float(label_j[label_j.find(':')+1:]))

                    fa -= 155
                    fb -= 155
                    # print(fa,fb)
                    for pp in range(fa,fb):
                        binary_label[pp] = 1

                ''' 第一个位置的噪声 '''
                x163_164 = [93,94]
                x163_165 = np.hstack((np.arange(68,93),np.array([41,95,96,97,98])))
                x164_166 = [3,5,6,7,8,9,10]
                x165_166 = [2,4]
                # 最多的 164-165
                if num in x163_164:
                    if binary_label[163-155] != 1:
                        binary_label[163-155] =2
                elif num in x163_165:
                    if binary_label[163-155] != 1:
                        binary_label[163-155] =2
                    if binary_label[164-155] != 1:
                        binary_label[164-155] =2
                elif num in x164_166:
                    if binary_label[164-155] != 1:
                        binary_label[164-155] =2
                    if binary_label[165-155] != 1:
                        binary_label[165-155] =2
                elif num in x165_166:
                    if binary_label[165-155] != 1:
                        binary_label[165-155] =2
                else:
                    if binary_label[164-155] != 1:
                        binary_label[164-155] =2

                ''' 第二个位置的噪声167-168,没有争议,如果那个位置没有电源,就认为是噪声'''
                # if binary_label[167-155] != 1:
                #     binary_label[167-155] = 2
                

                ''' 第三个位置的噪声 , 大多数是228-229 '''
                x229_230 = [2,3,4,5,6,7,8,9,10]
                if num in x229_230:
                    if binary_label[229-155]!=1:
                        binary_label[229-155]=2
                else:
                    if binary_label[228-155]!=1:
                        binary_label[228-155]=2

                ''' 最后一个位置的噪声230-231,,没有争议,如果那个位置没有电源,就认为是噪声'''
                # if binary_label[230-155] != 1:
                #     binary_label[230-155] = 2

                keep = path_i[path_i.find('_used_slicing')+14:-4]
                keepa = 155000. + int(keep[0:keep.find('xxx')])  # 184954.0
                keepb = 155000. + int(keep[keep.find('xxx')+3:]) # 244954.0


                for qq in binary_label:
                    if qq<=math.floor(keepa/1000-155) or qq>=math.ceil(keepb/1000-155):
                        binary_label[qq] = 0
                self.label_dir.append(binary_label)
                doing += 1
                print(f'{doing}/{total_num}')
                continue



            ''' 保护可以运行的, 之前产生二进制label的 方法'''
            # 将label转化为一串二进制
            # binary_label = []
            # loc = 0
            # while loc + self.sub_len <= len(frequency_data):
            #     flag = 0
            #     for j in data_label:
            #         if j >= self.begin+loc and j <= self.begin+loc+self.sub_len:
            #             flag = 1
            #             binary_label.append(1)
            #             break
            #     if flag == 0:
            #         binary_label.append(0)
            #     loc += self.sub_len
            ''' '''


    def __len__(self):
        return len(self.data_dir)
    
    def __getitem__(self,index):
        return self.data_dir[index],self.label_dir[index]








''' 保护原来可以用的三电源数据封装 '''
# class hjw_PowerDataset(Dataset):
#     def __init__(self,mode) -> None:
#         super().__init__()
#         self.label_path = r'D:\毕业设计\数据采集平台\label.txt'
#         if mode == 'train':
#             self.data_path = r'D:\毕业设计\数据采集平台\new_train_data++\data'
#         elif mode == 'validation':
#             self.data_path = r'D:\毕业设计\数据采集平台\new_val_data\data'
#         elif mode == 'test':
#             self.data_path = r'D:\毕业设计\数据采集平台\new_test_data\data'
#         else:
#             raise Exception('Please give a right mode.')

#         # 原始数据截取的总的频域区间
#         self.begin = begin
#         self.end = end
#         # 每个小区间长度为1000
#         self.sub_len = sub_len

#         self.mode = mode
#         self.data_dir = []
#         self.label_dir = []

#         with open(self.label_path) as f:
#             raw_label = f.readlines()

#         for i in range(len(os.listdir(self.data_path))):
#             full_path = os.path.join(self.data_path,os.listdir(self.data_path)[i])
#             num = int(os.listdir(self.data_path)[i][0:-4])


#             frequency_data = scipy.io.loadmat(full_path)['b']  # 频域区间130000 - 270000 ，长度140000                      
#             frequency_data = frequency_data[25000:-25000]  # 频域区间155000 - 245000 ，长度90000
#             # numpy数据类型
#             self.data_dir.append(frequency_data)



#             # # 2个或3个数值，记录中心频域的位置
#             data_label = raw_label[num-1].split(',')
#             data_label = [ float(j)*1000 for j in data_label]
#             # print(data_label)  # [181800.0, 231500.0]



#             # xx = np.arange(155000,245000)
#             # plt.plot(xx,frequency_data)
#             # plt.xticks(range(155000,245000,3000),range(155,245,3))
#             # plt.show()

#             ''' 基于NMS的产生二进制label的方法 '''
#             binary_label = [0]*90
#             for loc in data_label:
#                 kk =  (loc - self.begin)/self.sub_len
#                 xiaoshu, zhengshu = math.modf(kk)

#                 zhengshu = int(zhengshu)
#                 binary_label[zhengshu] = 1
#                 if xiaoshu >0.7:
#                     binary_label[zhengshu+1] = 1
#                 elif xiaoshu < 0.3:
#                     binary_label[zhengshu-1] = 1


#             ''' 保护可以运行的, 之前产生二进制label的 方法'''
#             # 将label转化为一串二进制
#             # binary_label = []
#             # loc = 0
#             # while loc + self.sub_len <= len(frequency_data):
#             #     flag = 0
#             #     for j in data_label:
#             #         if j >= self.begin+loc and j <= self.begin+loc+self.sub_len:
#             #             flag = 1
#             #             binary_label.append(1)
#             #             break
#             #     if flag == 0:
#             #         binary_label.append(0)
#             #     loc += self.sub_len
#             ''' '''

#             self.label_dir.append(binary_label)
#             # print(full_path)
#             # x = np.arange(130000,270000)
#             # plt.plot(x,frequency_data)
#             # plt.show()
#             # print(' ')



#     def __len__(self):
#         return len(self.data_dir)
    
#     def __getitem__(self,index):
#         return self.data_dir[index],self.label_dir[index]

