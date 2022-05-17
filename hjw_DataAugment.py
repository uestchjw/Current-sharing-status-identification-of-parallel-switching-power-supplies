
import os 
import shutil


import numpy as np
import scipy.io

import matplotlib.pyplot as plt
import matplotlib.patches as patches
train_data_path = r'D:\毕业设计\数据采集平台\new_train_data++\data'









# train_data_path = r'D:\毕业设计\数据采集平台\new_val_data++\data'
# for path_i in os.listdir(train_data_path):
#     if path_i.find('used') == -1:
#         full_path = os.path.join(train_data_path , path_i)
#         num = int(path_i[0:-4])
#         data = scipy.io.loadmat(full_path)['b'][25000:-25000] # data = (90000, 1)
#         # min_v = np.min(data)
#         # max_v = np.max(data)
#         # data = [ (i-min_v)/(max_v-min_v) for i in data]
#         name = full_path[0:-4]
#         x = np.arange(155000,245000)
#         plt.plot(x,data,linewidth = 3)
#         plt.axis('off')
#         plt.show()
#         #plt.plot(x,train_loss,label = 'Train_loss', linestyle = '--',color = 'b',linewidth = 2.8,marker = 'o',markersize = 13)
#         fig,axes = plt.subplots()


#         # y = np.arange(230000,233000)
#         # t = data[75000:78000]
#         # plt.plot(y,t)
#         # plt.show()
#         # np.save('D:/NMS样本1.npy',t)
#         # plt.figure(figsize=(14,8))
#         fig,axes = plt.subplots()
#         currentAxis = plt.gca()
#         num = 155000
#         for kk in range(90):
#             rect = patches.Rectangle((num,0),1000,0.4,linestyle = 'dotted',edgecolor = 'r',facecolor = 'none')
#             currentAxis.add_patch(rect)
#             num += 1000
#         plt.title(path_i)
#         font1 = {'family':'Times New Roman','size':28}

#         plt.plot(x,data)
#         plt.tick_params(labelsize = 28)
#         plt.xlabel('f(Hz)',font1)
#         plt.ylabel('Amplitude',font1)
#         x_kedu = axes.get_xticklabels()
#         [i.set_fontname('Times New Roman') for i in x_kedu]
#         y_kedu = axes.get_yticklabels()
#         [i.set_fontname('Times New Roman') for i in y_kedu]
#         # plt.xticks(np.arange(155000,246000,1000),np.arange(155,246,1))
#         plt.show()
#         # plt.savefig('D:/毕业设计/数据采集平台/训练数据图像/' + path_i[0:-4] +'.jpg',dpi = 200)
#         plt.close()










''' fake noise增强. 人工添加那些固定频率位置、容易被识别为中心频率的高幅值噪声 '''
def fakenoise():
    import random
    train_data_path = r'D:\毕业设计\数据采集平台\new_train_data++\data'
    for path_i in os.listdir(train_data_path):
        if path_i.find('used') == -1:
            full_path = os.path.join(train_data_path , path_i)
            data = scipy.io.loadmat(full_path)['b'][25000:-25000] # data = (90000, 1)
            name = full_path[0:-4]
            max_value = np.max(data)
            min_value = np.min(data)
            data = [(i-min_value)/(max_value-min_value) for i in data]
            # noise1 = scipy.io.loadmat(r'D:\毕业设计\数据采集平台\new_train_data++\noise1.mat')['b']
            # noise2 = scipy.io.loadmat(r'D:\毕业设计\数据采集平台\new_train_data++\noise2.mat')['b']
            # new_data = data
            # start = random.randint(0,89000)
            # new_data[start:start+1000] = noise2

            # x = np.arange(155000,245000)
            # plt.plot(x,new_data)
            # plt.show()
            # scipy.io.savemat(name+'_used_fakenoise2.mat',{'b':new_data})


            ''' 可视化操作 '''
            x = np.arange(155000,245000)
            x1 = x[26770:26800]
            signal = data[26770:26800]
            
            noise = data[26773:26780]

            # print(noise)
            x2 = x1
            signal2 = signal.copy()
            signal2[20:27] = noise
            # print(signal.flatten())
            # print(signal2.flatten())
            fig,axes = plt.subplots()

            font1 = {'family':'Times New Roman','size':32}
            font2 = {'family':'SimHei','size':42}

            plt.xlabel('f(KHz)',font1)
            plt.ylabel('Amplitude',font1)

            plt.tick_params(labelsize = 30)
            x_kedu = axes.get_xticklabels()
            [i.set_fontname('Times New Roman') for i in x_kedu]
            y_kedu = axes.get_yticklabels()
            [i.set_fontname('Times New Roman') for i in y_kedu]
            # axes.plot(x,signal,x,new_data,'r')
            plt.plot(x1,signal,label = '原始数据', linewidth = 2.5)
            plt.plot(x2,signal2,'r',label = '增强数据',linewidth = 2.5,color = 'r')
            plt.legend(prop = font2)
            plt.xticks([181770,181780,181790,181800],[181.77,181.78,181.79,181.8])

            # plt.scatter(x[26760:26800],data[26760:26800],x[26760:26800],new_data[26760:26800])
            plt.show()
            print(' ')

''' 窗口切片,只保留原来的2/3 ,剩下位置补0 '''
def window_slicing():
    import random
    import copy
    train_data_path = r'D:\毕业设计\数据采集平台\new_train_data++\data'
    for path_i in os.listdir(train_data_path):
        full_path = os.path.join(train_data_path , path_i)
        if path_i.find('used') == -1:
            data = scipy.io.loadmat(full_path)['b'][25000:-25000] # data = (90000, 1)
            max_value = np.max(data)
            min_value = np.min(data)
            data = [(i-min_value)/(max_value-min_value) for i in data]
            name = full_path[0:-4]
            new_data = []

            k = 2/3
            start = random.randint(0,int(len(data)*(1-k)))
            end = start+int(k*len(data))
            new_data = np.array(data.copy())
            new_data[0:start] = 0
            new_data[end:] = 0

            new_file = name + '_used_slicing_' + str(start) + 'xxx' + str(end) + '.mat'
            # print(new_file)  # D:\毕业设计\数据采集平台\new_train_data++\data\101_used_slicing_13080:73080.mat
            # scipy.io.savemat(new_file,{'b':new_data})
            # x = np.arange(155000,245000)
            # plt.plot(x,data,x,new_data)
            # plt.show()

            ''' 可视化操作 '''
            x = np.arange(155000,245000)
            x = x[26770:26800]
            signal = data[26770:26800]
            k = 2/3
            start = random.randint(0,int(len(signal)*(1-k)))
            end = start+int(k*len(signal))
            print(start,end)
            new_data = np.array(signal.copy())
            new_data[0:start] = 0
            new_data[end:] = 0

            fig,axes = plt.subplots()

            font1 = {'family':'Times New Roman','size':32}
            font2 = {'family':'SimHei','size':42}

            plt.xlabel('f(KHz)',font1)
            plt.ylabel('Amplitude',font1)

            plt.tick_params(labelsize = 32)
            x_kedu = axes.get_xticklabels()
            [i.set_fontname('Times New Roman') for i in x_kedu]
            y_kedu = axes.get_yticklabels()
            [i.set_fontname('Times New Roman') for i in y_kedu]
            # axes.plot(x,signal,x,new_data,'r')
            plt.plot(x,signal,label = '原始数据',linewidth = 2.5)
            plt.plot(x,new_data,'r',label = '增强数据',linewidth = 2.5, color = 'red')
            plt.legend(prop = font2)
            plt.xticks([181770,181780,181790,181800],[181.77,181.78,181.79,181.8])

            # plt.scatter(x[26760:26800],data[26760:26800],x[26760:26800],new_data[26760:26800])
            plt.show()
            print(' ')

''' random噪声增强 '''
def randomnoise():
    import random
    train_data_path = r'D:\毕业设计\数据采集平台\new_train_data++\data'
    for path_i in os.listdir(train_data_path):
        full_path = os.path.join(train_data_path , path_i)
        if path_i.find('used') == -1:
            data = scipy.io.loadmat(full_path)['b'][25000:-25000] # data = (90000, 1)
            max_value = np.max(data)
            min_value = np.min(data)
            data = [(i-min_value)/(max_value-min_value) for i in data]
            name = full_path[0:-4]
            # print(name)  # D:\毕业设计\数据采集平台\new_train_data++\data\101
            new_data = []
            for k in data:
                random_noise = random.uniform(-0.3*k,0.3*k)
                if k + random_noise >=0:
                    new_data.append(k+random_noise)
                else:
                    new_data.append(0)

            x = np.arange(155000,245000)
            fig,axes = plt.subplots()
            font1 = {'family':'Times New Roman','size':32}
            font2 = {'family':'SimHei','size':42}
            plt.xlabel('f(KHz)',font1)
            plt.ylabel('Amplitude',font1)

            plt.tick_params(labelsize = 32)
            x_kedu = axes.get_xticklabels()
            [i.set_fontname('Times New Roman') for i in x_kedu]
            y_kedu = axes.get_yticklabels()
            [i.set_fontname('Times New Roman') for i in y_kedu]
            plt.plot(x[26770:26800],data[26770:26800],linewidth = 2.5,label = '原始数据')
            plt.plot(x[26770:26800],new_data[26770:26800],color= 'r',linewidth = 2.5,label = '增强数据')
            plt.xticks([181770,181780,181790,181800],[181.77,181.78,181.79,181.8])

            plt.legend(prop = font2)
            plt.show()
            scipy.io.savemat(name+'_used_randomnoise.mat',{'b':new_data})

''' scale增强, 不改变label, 产生的D:\毕业设计\数据采集平台\new_train_data++\data\101_used_scale.mat 还在该文件夹里面 '''
def scale():
    train_data_path = r'D:\毕业设计\数据采集平台\new_train_data++\data'
    for path_i in os.listdir(train_data_path):
        full_path = os.path.join(train_data_path , path_i)
        if path_i.find('used') == -1:
            name = full_path[0:-4]
            data = scipy.io.loadmat(full_path)['b'][25000:-25000] # data = (90000, 1)
            max_value = np.max(data)
            min_value = np.min(data)
            data = [(i-min_value)/(max_value-min_value) for i in data]
            new_data = []
            for value in data:
                if value > 0.3*max_value:
                    new_data.append(0.85*value)
                else:
                    new_data.append(1.15*value)
            
            x = np.arange(155000,245000)
            # max_value = np.max(new_data)
            # min_value = np.min(new_data)
            # new_data = [(i-min_value)/(max_value-min_value) for i in new_data]

            # print(x[26770:26800])
            # [181770 181771 181772 181773 181774 181775 181776 181777 181778 181779
            # 181780 181781 181782 181783 181784 181785 181786 181787 181788 181789
            # 181790 181791 181792 181793 181794 181795 181796 181797 181798 181799]
            fig,axes = plt.subplots()
            font1 = {'family':'Times New Roman','size':32}
            font2 = {'family':'SimHei','size':40}
            plt.xlabel('f(KHz)',font1)
            plt.ylabel('Amplitude',font1)

            plt.tick_params(labelsize = 32)
            x_kedu = axes.get_xticklabels()
            [i.set_fontname('Times New Roman') for i in x_kedu]
            y_kedu = axes.get_yticklabels()
            [i.set_fontname('Times New Roman') for i in y_kedu]
            plt.plot(x[26770:26800],data[26770:26800],linewidth = 2.5,label = '原始数据')
            plt.plot(x[26770:26800],new_data[26770:26800],color= 'r',linewidth = 2.5,label = '增强数据')
            
            plt.xticks([181770,181780,181790,181800],[181.77,181.78,181.79,181.8])

            plt.legend(prop = font2)
            plt.show()
            # scipy.io.savemat(name+'_used_scale.mat',{'b':new_data})


if __name__ == '__main__':
    # window_slicing()
    fakenoise()
    randomnoise()
    scale()
    




