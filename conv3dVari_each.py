import torch
import torch.nn as nn
import torch.utils.data as data_utl
import os
from PIL import Image
import numpy as np
import random
import pandas as pd

def video_to_tensor(pic):
    """
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))

def load_rgb_frames(image_dir, vid, f_ind, resize=None):#resize=(H, W)
  frames = []

  for i in f_ind:

    fname = os.path.join(image_dir,
                         str(vid).zfill(4), str(vid).zfill(4)+'_'+str(i).zfill(5)+'.jpg')

    img = Image.open(fname).convert('RGB')

    ########################################################################
    # resize images
    ########################################################################
    if resize is not None: img = img.resize(size=resize, resample=Image.LANCZOS)
    img=np.array(img)


    if len(f_ind)==1:#when modelName == 'vgg'
        # normalize image between 0 and 1
        img = (img-np.min(img))/(np.max(img)-np.min(img))
    else:
    ########################################################################
    # normalize image between -1 and 1 #np.max(img)==255 #
    ########################################################################
        #img = (img/255.)*2 - 1
        img = (img/np.max(img))*2 - 1

    frames.append(img)

  return np.asarray(frames, dtype=np.float32)

def load_flow_frames(image_dir, vid, f_ind, resize=None):
  frames = []
  for i in f_ind:
    fx = os.path.join(image_dir,str(vid).zfill(4), str(vid).zfill(4)+'_'+str(i).zfill(5)+'x.jpg')
    fy = os.path.join(image_dir,str(vid).zfill(4), str(vid).zfill(4)+'_'+str(i).zfill(5)+'y.jpg')
    
    if fx is not None:
        imgx = Image.open(fx).convert('L')
        imgy = Image.open(fy).convert('L')
        
        # w,h = imgx.shape
        # if w < 224 or h < 224:
        #     d = 224.-min(w,h)
        #     sc = 1+d/min(w,h)
        #     imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
        #     imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
        if resize is not None: 
            imgx = imgx.resize(size=resize, resample=Image.LANCZOS)
            imgy = imgy.resize(size=resize, resample=Image.LANCZOS)
        imgx = np.array(imgx)
        imgy = np.array(imgy)   

        if len(f_ind)==1:
            imgx = (imgx-np.min(imgx))/(np.max(imgx)-np.min(imgx))
            imgy = (imgy-np.min(imgy))/(np.max(imgy)-np.min(imgy))
        else:
            imgx = (imgx/np.max(imgx))*2 - 1
            imgy = (imgy/np.max(imgy))*2 - 1
        img = np.asarray([imgx, imgy]).transpose([1,2,0])
        frames.append(img)
    else:
        print('no flow!')    
  return np.asarray(frames, dtype=np.float32)

def make_label_array(f_ind, SL, EL, num_classes):
    label = np.zeros((num_classes, len(f_ind)), np.float32)
    for i in range(len(f_ind)):
        if SL <= f_ind[i] <= EL:
            label[1, i] = 1
        else:
            label[0, i] = 1
    return label

def make_dataset(split_file, split, root, num_classes):
    dataset = []

    df = pd.read_excel(split_file)
    
    idx= df.loc[df['split']==split].index
    #num_classes=2로 설계돼있음
    for i in idx:
      num_frame = df.loc[i, 'num_frame']
      start_frame = df.loc[i, 'start_frame']
      end_frame = df.loc[i, 'end_frame']

      lvc_start = min(df.loc[i, 'label_start'], df.loc[i, 'label_end'])
      lvc_end = max(df.loc[i, 'label_start'], df.loc[i, 'label_end']) 

      vid = df.loc[i, 'patient_id']
      
      if not os.path.exists(os.path.join(root, str(vid).zfill(4))):
        print('no video!')
        continue

    #   label = np.zeros((num_classes, num_frame), np.float32)
    #   label[1, lvc_start-start_frame:lvc_end-start_frame+1] = 1
    #   label[0, :] = 1 - label[1, :]

      dataset.append((vid, start_frame, end_frame, num_frame, lvc_start, lvc_end))#, label
    
    return dataset

def calIoU(f_ind, SL, EL):# how much of GT is contained? the input must not have duplicates
    sum=0
    for i in f_ind:
        if i in range(SL, EL+1):
            sum+=1
    return sum/(EL+1-SL)

class VFSS_data(data_utl.Dataset):

    def __init__(self, split_file, split, root, frame_len, sampling, 
    transforms, input_type='rgb', num_classes=2, resize=None, load_whole= False):
        #if uniform, randint within the label range

        self.split_file = split_file
        self.transforms = transforms
        self.root = root
        self.frame_len = frame_len
        self.load_whole = load_whole
        self.resize = resize
        self.sampling = sampling
        self.split=split
        self.num_classes = num_classes
        self.input_type = input_type

        self.data = make_dataset(split_file, split, root, num_classes) 

        # print(len(self.data))#180

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            video_to_tensor(imgs), torch.from_numpy(label), IoU, vid

        if self.seglabel, random(0,1) if 1, sample 16 within positive label region             
        """
        vid, start_frame, end_frame, num_frame, SL, EL= self.data[index]#, label

        # with some probability, include at least some part of the 
        if self.load_whole:
          f_ind = list(range(start_frame, end_frame+1))

        elif self.frame_len==1: #make positive sample one third!!
            coin = random.randint(0,2)
            if coin==2:
                positive = list(range(SL, EL+1))
                f_ind = [random.choice(positive)]
            else:
                negative = list(range(start_frame, SL))+list(range(EL+1, end_frame+1))
                f_ind = [random.choice(negative)]

        elif self.sampling[0] == 'p':
            flen = self.frame_len
            if self.split == 'train':
                coin = random.randint(0,9)
                cth = 3
                if coin < cth: #if it should be a positive label
                    offset = random.randint(0,8)
                    start_f = min(max(SL - start_frame - offset,1), num_frame-flen)
                else:#if it should be a background label
                    start_f = random.randint(1,num_frame-flen)   
            else:
                start_f = random.randint(1,num_frame-flen)

            f_ind=list(range(start_frame+start_f, start_frame+start_f+flen))

        elif self.sampling[0] == 'o':
            total = np.array(range(start_frame, end_frame+1))
            posi_len = int(self.frame_len/3)
            positive = np.array(range(SL, EL+1))#this is always positive
            # print(len(total), posi_len, len(positive))
            if len(positive) < posi_len:#positive 갯수만큼 positive frame 수 중복으로 늘리기
                posi_f_ind = np.sort(np.random.randint(SL, EL+1, posi_len-len(positive)))#positive 갯수만큼 positive frame 수 중복으로 늘리기
                total = np.sort(np.concatenate((total, posi_f_ind)))#늘린 frame들 추가하기
            neg_s_len = self.frame_len - posi_len
            neg_e_len = self.frame_len - posi_len
            ns = np.array(range(start_frame, SL))
            if ns.size==0: neg_s_len = 0
            elif len(ns)<neg_s_len:#negative 갯수만큼 SL이전 negative갯수 늘리기
                ns_f_ind = np.sort(np.random.randint(start_frame, SL, neg_s_len-len(ns)))
                total = np.sort(np.concatenate((total, ns_f_ind)))#늘린 frame들 추가하기
            ne = np.array(range(EL+1, end_frame+1))
            if ne.size==0: neg_e_len = 0
            elif len(ne)<neg_e_len:#negative 갯수만큼 EL이후 negative갯수 늘리기
                ne_f_ind = np.sort(np.random.randint(EL+1, end_frame+1, neg_e_len-len(ne)))
                total = np.sort(np.concatenate((total, ne_f_ind)))#늘린 frame들 추가하기

            tot_posi_s_ind = np.where(SL<=total)
            tot_posi_e_ind = np.where(total<=EL)
            tot_posi_ind = np.intersect1d(tot_posi_s_ind, tot_posi_e_ind)

            if neg_s_len == 0:
                if neg_e_len == 0:
                    start_f = random.choice(range(min(tot_posi_ind), max(tot_posi_ind)-self.frame_len))
                else:
                    start_f = random.choice(range(min(tot_posi_ind), max(tot_posi_ind)- posi_len))
            elif neg_e_len == 0:# when neg_s_len != 0
                start_f = random.choice(range(min(tot_posi_ind)-neg_s_len, max(tot_posi_ind) - self.frame_len))
            else:# when neg_s_len, neg_e_len != 0
                start_f_ind = list(range(min(tot_posi_ind)-neg_s_len-1, max(tot_posi_ind)-posi_len+1))
                start_f = random.choice(start_f_ind)
            f_ind_ind = list(range(start_f, start_f+self.frame_len))
            f_ind = total[f_ind_ind]

        else:#추출을 원하는 구간이 self.frame_len 보다 작은 경우도 생각..!
            whether=np.random.randint(0, 2, 1)
            if whether: #IoU가 1이 되도록..!
                if self.sampling[0] == 'u': f_ind = np.sort(np.random.randint(SL, EL+1, self.frame_len))
                elif self.sampling[0] == 's':
                    if EL+1-self.frame_len <= SL:#추출을 원하는 구간이 self.frame_len 보다 작은 경우
                        f_ind=list(np.sort(np.random.randint(SL, EL+1, self.frame_len)))
                        
                    else:
                        start=np.random.randint(SL, EL+1-self.frame_len, 1)[0]
                        f_ind=list(range(start, start+self.frame_len))
                
            else:#IoU가 0이 되도록..!
                if self.sampling[0] == 'u':
                    f_ind=np.sort(np.random.choice(list(range(start_frame, SL))+list(range(EL+1, end_frame+1)), self.frame_len))
                elif self.sampling[0] == 's':
                    neg=list(range(start_frame, SL))+list(range(EL+1, end_frame+1))
                    if len(neg)<self.frame_len:#추출을 원하는 구간이 self.frame_len 보다 작은 경우
                        f_ind=list(np.sort(np.random.choice(neg, self.frame_len)))
                    else:
                        start_index=len(neg)-self.frame_len
                        f_ind=neg[start_index:]
                
        IoU=calIoU(set(f_ind), SL, EL)    
        IoU_TH = self.frame_len / len(range(start_frame, end_frame+1))
        iouLabel = 1 if IoU >= IoU_TH else 0

        if self.load_whole: 
            return torch.tensor(f_ind, dtype=torch.int32), SL, EL, iouLabel, vid

        label = make_label_array(f_ind, SL, EL, self.num_classes)
        
        if self.input_type=='rgb':
            imgs = load_rgb_frames(self.root, vid, f_ind, self.resize)
        elif self.input_type=='flow':
            imgs = load_flow_frames(self.root, vid, f_ind, self.resize)

        imgs = video_to_tensor(imgs)
        if self.transforms is not None:
            imgs = imgs.transpose(0,1)
            imgs = self.transforms(imgs)
            imgs = imgs.transpose(0,1)

        return imgs, torch.from_numpy(label), iouLabel, vid

    def __len__(self):
         return len(self.data)

#The effect of time on the automated detection of the pharyngeal phase
#in videofluoroscopic swallowing studies
class CNN3D(nn.Module): #(frames=8,224,224,channel=1(gray scale))
    def __init__(self, num_classes=2):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1_1 = nn.Conv3d(1, 4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv1_2 = nn.Conv3d(4, 4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2_1 = nn.Conv3d(4, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2_2 = nn.Conv3d(8, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3_1 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3_2 = nn.Conv3d(16, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4_1 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4_2 = nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.last_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.fc1 = nn.Linear(32*14*14, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)
    def forward(self, x): 
        # print(x.size()) #[batch, channel=1, frame_num=8, H=224, W=224]
        x=self.relu(self.conv1_1(x)) 
        # print(x.size()) #[batch, channel=4, frame_num=8, H=224, W=224]
        x=self.relu(self.conv1_2(x)) 
        # print(x.size()) #[batch, channel=4, frame_num=8, H=224, W=224]
        x=self.pool(x) 
        # print(x.size()) #[batch, channel=4, frame_num=4, H=112, W=112]
        x=self.relu(self.conv2_1(x)) 
        # print(x.size()) #[batch, channel=8, frame_num=4, H=112, W=112]
        x=self.relu(self.conv2_2(x))
        # print(x.size()) #[batch, channel=8, frame_num=4, H=112, W=112]
        x=self.pool(x)
        # print(x.size()) #[batch, channel=8, frame_num=2, H=56, W=56]
        x=self.relu(self.conv3_1(x))
        # print(x.size()) #[batch, channel=16, frame_num=2, H=56, W=56]
        x=self.relu(self.conv3_2(x))
        # print(x.size()) #[batch, channel=16, frame_num=2, H=56, W=56]
        x=self.pool(x)
        # print(x.size()) #[batch, channel=16, frame_num=1, H=28, W=28]
        x=self.relu(self.conv4_1(x))
        # print(x.size()) #[batch, channel=32, frame_num=1, H=28, W=28]
        x=self.relu(self.conv4_2(x))
        # print(x.size()) #[batch, channel=32, frame_num=1, H=28, W=28]
        x=self.last_pool(x)
        # print(x.size()) #[batch, channel=32, frame_num=1, H=14, W=14]
        x=x.view(x.size(0), -1)
        # print(x.size()) #[batch, 32*14*14=6272]
        x=self.relu(self.fc1(x))
        # print(x.size()) #[batch, 128]
        x=self.relu(self.fc2(x))
        # print(x.size()) #[batch, 64]
        x=self.out(x)
        # print(x.size()) #[batch, 2]
        return x


##############################################################################################################
#(H=512, W=512)
class cnn3d2(nn.Module):#layer 15개 중 conv 6개
    def __init__(self, num_classes=2, lin_in=2048, lin_out=1024, input_type='rgb'):
        super().__init__()

        if input_type=='rgb': channel=3
        else: channel=2

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv3d(channel, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2a = nn.Conv3d(64, 128, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.conv2b = nn.Conv3d(128, 128, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.fc9 = nn.Linear(lin_in, lin_out)
        self.fc10 = nn.Linear(lin_out, lin_out)
        self.fc11 = nn.Linear(lin_out, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.__init_weight()


    def forward(self, x):
        #print(x.size())# [batch, RGB=3, frame_num=2, H, W]

        x = self.relu(self.conv1(x))
        # print(x.size())# [batch, 64, frame_num=2, H, W]
        x = self.pool1(x)
        # print(x.size())# [batch, 64, frame_num=2, H/2, W/2]

        x = self.relu(self.conv2a(x))
        # print(x.size())# [batch, 128, frame_num=2, H/4, W/4]
        x = self.relu(self.conv2b(x))
        # print(x.size())# [batch, 128, frame_num=2, H/8, W/8]
        x = self.pool2(x)  
        # print(x.size())# [batch, 128, frame_num=2, H/16, W/16] 

        x = self.relu(self.conv3(x))
        # print(x.size())# [batch, 256, frame_num=2, H/16, W/16]
        x = self.pool3(x) # x = self.relu(self.conv4(x))
        # print(x.size())# [batch, 256, frame_num=1, H/32, W/32] 

        x = self.relu(self.conv4a(x))
        # print(x.size())# [batch, 512, frame_num=1, H/64, W/64] 
        x = self.relu(self.conv4b(x)) # x = self.relu(self.conv6(x))
        # print(x.size())# [batch, 512, frame_num=1, H/128, W/128] !!!!!!!!!
        x = self.pool4(x) # x = self.relu(self.conv7(x))
        # print(x.size())# [batch, 512, frame_num=1, H/256, W/256] 
        
        x = x.view(x.size(0), -1)
        # print(x.size())# [batch, 2048]
        x = self.relu(self.fc9(x))
        # print(x.size())# [batch, 1024]
        x = self.dropout(x)
        x = self.relu(self.fc10(x))
        x = self.dropout(x)
        x = self.fc11(x)

        return x # [batch, 2]

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class cnn3d4(nn.Module):#layer 15개 중 conv 6개
    def __init__(self, num_classes=2, lin_in=2048, lin_out=1024, input_type='rgb'):
        super().__init__()

        if input_type=='rgb': channel=3
        else: channel=2

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv3d(channel, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2a = nn.Conv3d(64, 128, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.conv2b = nn.Conv3d(128, 128, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc9 = nn.Linear(lin_in, lin_out)
        self.fc10 = nn.Linear(lin_out, lin_out)
        self.fc11 = nn.Linear(lin_out, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.__init_weight()

    def forward(self, x):
        #print(x.size())# [batch, RGB=3, frame_num=4, H, W]

        x = self.relu(self.conv1(x))
        # print(x.size())# [batch, 64, frame_num=4, H, W]
        x = self.pool1(x)
        # print(x.size())# [batch, 64, frame_num=4, H/2, W/2]

        x = self.relu(self.conv2a(x))
        # print(x.size())# [batch, 128, frame_num=4, H/4, W/4]
        x = self.relu(self.conv2b(x))
        # print(x.size())# [batch, 128, frame_num=4, H/8, W/8]
        x = self.pool2(x)  
        # print(x.size())# [batch, 128, frame_num=2, H/16, W/16] 

        x = self.relu(self.conv3(x))
        # print(x.size())# [batch, 256, frame_num=2, H/16, W/16]
        x = self.pool3(x) # x = self.relu(self.conv4(x))
        # print(x.size())# [batch, 256, frame_num=1, H/32, W/32] 

        x = self.relu(self.conv4a(x))
        # print(x.size())# [batch, 512, frame_num=1, H/64, W/64] 
        x = self.relu(self.conv4b(x)) # x = self.relu(self.conv6(x))
        # print(x.size())# [batch, 512, frame_num=1, H/128, W/128] !!!!!!!!!
        x = self.pool4(x) # x = self.relu(self.conv7(x))
        # print(x.size())# [batch, 512, frame_num=1, H/256, W/256] 
        
        x = x.view(x.size(0), -1)
        # print(x.size())# [batch, 2048]
        x = self.relu(self.fc9(x))
        # print(x.size())# [batch, 1024]
        x = self.dropout(x)
        x = self.relu(self.fc10(x))
        x = self.dropout(x)
        x = self.fc11(x)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class cnn3d8(nn.Module):# layer 15개 중 conv 8개
    def __init__(self, num_classes=2, lin_in=2048, lin_out=1024, input_type='rgb'):
        super().__init__()

        if input_type=='rgb': channel=3
        else: channel=2

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv3d(channel, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # self.conv5a = nn.Conv3d(512, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        # self.conv5b = nn.Conv3d(512, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        # self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))#???

        self.fc9 = nn.Linear(lin_in, lin_out)
        self.fc10 = nn.Linear(lin_out, lin_out)
        self.fc11 = nn.Linear(lin_out, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.__init_weight()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        # print(x.size())# torch.Size([batch, 64, frame_num=8, W/2, W/2])
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        # print(x.size())# torch.Size([batch, 128, 4, W/4, W/4])
        x = self.relu(self.conv3a(x))
        # print(x.size())# torch.Size([batch, 256, 4, W/8, W/8])
        x = self.relu(self.conv3b(x))
        # print(x.size())# torch.Size([batch, 256, 4, W/16, W/16])
        x = self.pool3(x)
        # print(x.size())# torch.Size([batch, 256, 2, W/32, W/32])
        x = self.relu(self.conv4a(x))
        # print(x.size())# torch.Size([batch, 512, 2, W/64, W/64])
        x = self.relu(self.conv4b(x))
        # print(x.size())# torch.Size([batch, 512, 2, W/128, W/128])
        x = self.pool4(x)
        # print(x.size())# torch.Size([batch, 512, 1, W/256, W/256])

        x = x.view(x.size(0), -1)
        # print(x.size())# [batch, 2048]
        x = self.relu(self.fc9(x))
        # print(x.size())# [batch, 1024]
        x = self.dropout(x)
        x = self.relu(self.fc10(x))
        x = self.dropout(x)
        x = self.fc11(x)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

#C3D 변형(112, 112) layer 18개 중 conv 8개###########################################################################
class C3D(nn.Module):
    def __init__(self, k2, k3, k4, k5=1, lin_in=8192, lin_out=4096, num_classes=2, input_type='rgb'):
        super(C3D, self).__init__()

        if input_type=='rgb': channel=3
        else: channel=2

        self.conv1 = nn.Conv3d(channel, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(k2, 2, 2), stride=(k2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(k3, 2, 2), stride=(k3, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(k4, 2, 2), stride=(k4, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(k5, 2, 2), stride=(k5, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(lin_in, lin_out)
        self.fc7 = nn.Linear(lin_out, lin_out)
        self.fc8 = nn.Linear(lin_out, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        # print(x.size())
        x = self.relu(self.conv5a(x))
        # print(x.size())
        x = self.relu(self.conv5b(x))
        # print(x.size())
        x = self.pool5(x)
        # print(x.size())

        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.relu(self.fc6(x))
        # print(x.size())  
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)

        return logits

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k
'''
if __name__ == "__main__":
    frame_len=1
    inputs = torch.zeros(1, 3, frame_len, 112, 112)
    if frame_len==8:
        k2, k3, k4 = 2, 2, 2
    elif frame_len==4:
        k2, k3, k4 = 2, 2, 1
    elif frame_len==2:
        k2, k3, k4 = 1, 2, 1
    elif frame_len==1:
        k2, k3, k4 = 1, 1, 1

    net = C3D(k2, k3, k4)

    outputs = net.forward(inputs)
'''
#VGG 3d로 변형 (224, 224) laye21개 중 conv 8개########################################################################################
class vgg3d(nn.Module):
    def __init__(self, num_classes=2, input_type='rgb'):
        super().__init__()

        if input_type=='rgb': channel=3
        else: channel=2

        self.relu = nn.ReLU()

        self.m=nn.MaxPool3d(kernel_size=(1, 2, 2), stride=2)

        self.c1_1=nn.Conv3d(channel, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.c1_2=nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        self.c2_1=nn.Conv3d(64, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.c2_2=nn.Conv3d(128, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        self.c3_1=nn.Conv3d(128, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.c3_23=nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        self.c4_1=nn.Conv3d(256, 512, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.c4_=nn.Conv3d(512, 512, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 7, 7))#output_size
        self.drop = nn.Dropout()

        self.fc1=nn.Linear(512*7*7, 4096)
        self.fc2=nn.Linear(4096, 4096)
        self.fc3=nn.Linear(4096, num_classes)

        self._initialize_weights()

    def forward(self, x):
        # print(x.size())#(batch, RGB, frame_len, H, W)
        x=self.relu(self.c1_1(x))
        # print(x.size())
        x=self.relu(self.c1_2(x))
        # print(x.size())
        x=self.m(x)
        # print(x.size())

        x=self.relu(self.c2_1(x))
        # print(x.size())
        x=self.relu(self.c2_2(x))
        # print(x.size())
        x=self.m(x)
        # print(x.size())

        x=self.relu(self.c3_1(x))
        # print(x.size())
        x=self.relu(self.c3_23(x))
        # print(x.size())
        x=self.relu(self.c3_23(x))
        # print(x.size())
        x=self.m(x)
        # print(x.size())

        x=self.relu(self.c4_1(x))
        # print(x.size())
        x=self.relu(self.c4_(x))
        # print(x.size())
        x=self.relu(self.c4_(x))
        # print(x.size())
        x=self.m(x)
        # print(x.size())

        x=self.relu(self.c4_(x))
        # print(x.size())
        x=self.relu(self.c4_(x))
        # print(x.size())
        x=self.relu(self.c4_(x))
        # print(x.size())
        x=self.m(x)
        # print(x.size())

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x=self.drop(self.relu(self.fc1(x)))
        # print(x.size())#torch.Size([1, 4096])
        x=self.drop(self.relu(self.fc2(x)))
        # print(x.size())#torch.Size([1, 4096])
        x=self.fc3(x)
        # print(x.size())#torch.Size([1, 2])

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

#cnn3d+frame 별 label (H=512, W=512)########################################################################################
class cnn3dframeLabel(nn.Module):#layer 15개 중 conv 6개 
    def __init__(self, num_classes=2, input_type='rgb'):
        super().__init__()

        if input_type=='rgb': channel=3
        else: channel=2

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv3d(channel, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2a = nn.Conv3d(64, 128, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.conv2b = nn.Conv3d(128, 128, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.linear0 = nn.Linear(512*2*2, 512)
        self.linear1 = nn.Linear(512, 128)
        self.linear2 = nn.Linear(128, num_classes)


        self.__init_weight()

    def forward(self, x):
        # print(x.size())# [batch, RGB=3, frame_num=4, H=512, W=512]
        # print(torch.isnan(torch.sum(x)).item(), end=' ')
        x = self.relu(self.conv1(x))
        # print(x.size())# [batch, 64, frame_num=4, H=512, W=512]
        # print(torch.isnan(torch.sum(x)).item(), end=' ')
        x = self.pool1(x)
        # print(x.size())# [batch, 64, frame_num=4, H/2=256, W/2=256]
        # print(torch.isnan(torch.sum(x)).item(), end=' ')

        x = self.relu(self.conv2a(x))
        # print(x.size())# [batch, 128, frame_num=4, H/4=128, W/4=128]
        # print(torch.isnan(torch.sum(x)).item(), end=' ')
        x = self.relu(self.conv2b(x))
        # print(x.size())# [batch, 128, frame_num=4, H/8=64, W/8=64]
        # print(torch.isnan(torch.sum(x)).item(), end=' ')
        x = self.pool2(x)  
        # print(x.size())# [batch, 128, frame_num=4, H/16=32, W/16=32] 
        # print(torch.isnan(torch.sum(x)).item(), end=' ')

        x = self.relu(self.conv3(x))
        # print(x.size())# [batch, 256, frame_num=4, H/16=32, W/16=32]
        # print(torch.isnan(torch.sum(x)).item(), end=' ')
        x = self.pool3(x) 
        # print(x.size())# [batch, 256, frame_num=1, H/32=16, W/32=16] 
        # print(torch.isnan(torch.sum(x)).item(), end=' ')

        x = self.relu(self.conv4a(x))
        # print(x.size())# [batch, 512, frame_num=4, H/64=8, W/64=8] 
        # print(torch.isnan(torch.sum(x)).item(), end=' ')
        x = self.relu(self.conv4b(x)) 
        # print(x.size())# ku[batch, 512, frame_num=4, H/128=4, W/128=4]
        # print(torch.isnan(torch.sum(x)).item(), end=' ') 
        x = self.pool4(x) 
        # print(x.size())# [batch, 512, frame_num=4, H/256=2, W/256=2]
        # print(torch.isnan(torch.sum(x)).item(), end=' ')
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # print(x.size())# [batch, 4, 512, 2, 2]
        # print(torch.isnan(torch.sum(x)).item(), end=' ')
        x = x.view(x.size(0), x.size(1),-1)
        # print(x.size())# [batch, frame_len=4, 2048] #512*2*2
        # print(torch.isnan(torch.sum(x)).item(), end=' ')

        x = self.linear0(x)
        # print(x.size())#torch.Size([10, 4, 512])
        x = self.linear1(x)
        # print(x.size())#torch.Size([10, 4, 128])
        x = self.linear2(x)
        # print(x.size())#torch.Size([10, 4, 2])
        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

#cnn3d+GRU (H=512, W=512)########################################################################################
class cnn3dGRU(nn.Module):#layer 15개 중 conv 6개 # LR 1e-4, 1e-2수렴안됨. 1e-3으로 학습

    def __init__(self, num_classes=2, input_type='rgb'):
        super().__init__()

        if input_type=='rgb': channel=3
        else: channel=2

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv3d(channel, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2a = nn.Conv3d(64, 128, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.conv2b = nn.Conv3d(128, 128, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.gru = nn.GRU(input_size=512*2*2, hidden_size=256, batch_first=True, bias=True, bidirectional=False)

        self.linear0 = nn.Linear(256, 128)
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, num_classes)

        self.__init_weight()

    def forward(self, x):
        # print(x.size())# [batch, RGB=3, frame_num=4, H=512, W=512]
        # print(torch.isnan(torch.sum(x)).item(), end=' ')
        x = self.relu(self.conv1(x))
        # print(x.size())# [batch, 64, frame_num=4, H=512, W=512]
        # print(torch.isnan(torch.sum(x)).item(), end=' ')
        x = self.pool1(x)
        # print(x.size())# [batch, 64, frame_num=4, H/2=256, W/2=256]
        # print(torch.isnan(torch.sum(x)).item(), end=' ')

        x = self.relu(self.conv2a(x))
        # print(x.size())# [batch, 128, frame_num=4, H/4=128, W/4=128]
        # print(torch.isnan(torch.sum(x)).item(), end=' ')
        x = self.relu(self.conv2b(x))
        # print(x.size())# [batch, 128, frame_num=4, H/8=64, W/8=64]
        # print(torch.isnan(torch.sum(x)).item(), end=' ')
        x = self.pool2(x)  
        # print(x.size())# [batch, 128, frame_num=4, H/16=32, W/16=32] 
        # print(torch.isnan(torch.sum(x)).item(), end=' ')

        x = self.relu(self.conv3(x))
        # print(x.size())# [batch, 256, frame_num=4, H/16=32, W/16=32]
        # print(torch.isnan(torch.sum(x)).item(), end=' ')
        x = self.pool3(x) 
        # print(x.size())# [batch, 256, frame_num=1, H/32=16, W/32=16] 
        # print(torch.isnan(torch.sum(x)).item(), end=' ')

        x = self.relu(self.conv4a(x))
        # print(x.size())# [batch, 512, frame_num=4, H/64=8, W/64=8] 
        # print(torch.isnan(torch.sum(x)).item(), end=' ')
        x = self.relu(self.conv4b(x)) 
        # print(x.size())# [batch, 512, frame_num=4, H/128=4, W/128=4]
        # print(torch.isnan(torch.sum(x)).item(), end=' ') 
        x = self.pool4(x) 
        # print(x.size())# [batch, 512, frame_num=4, H/256=2, W/256=2]
        # print(torch.isnan(torch.sum(x)).item(), end=' ')


        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # print(x.size())# [batch, f, 512, 2, 2]
        x = x.view(x.size(0), x.size(1),-1)
        # print(x.size())# [batch, f, 2048]
        x, hidden = self.gru(x, None)
        # print(x.size())# [batch, f, 256]

        x = self.linear0(x)
        # print(x.size())# [batch, f, 128]
        x = self.linear1(x)
        # print(x.size())# [batch, f, 64]
        x = self.linear2(x)
        # print(x.size())# [batch, f, num_classes=2]
        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

#cnn3d+GRU (H=512, W=512)########################################################################################
class cnn3dGRUbidirec2layer(nn.Module):#layer 15개 중 conv 6개 # batch1 일 때, LR 1e-4, 1e-2수렴안됨. 1e-3으로 학습

    def __init__(self, num_classes=2, input_type='rgb'):
        super().__init__()

        if input_type=='rgb': channel=3
        else: channel=2

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv3d(channel, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2a = nn.Conv3d(64, 128, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.conv2b = nn.Conv3d(128, 128, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.gru = nn.GRU(input_size=512*2*2, hidden_size=256, batch_first=True, bias=True, bidirectional=True, num_layers = 2)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.linear0 = nn.Linear(256, 128)
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, num_classes)

        self.__init_weight()

    def forward(self, x):
        # print(x.size())# [batch, RGB=3, frame_num=4, H=512, W=512]
        # print(torch.isnan(torch.sum(x)).item(), end=' ')
        x = self.relu(self.conv1(x))
        # print(x.size())# [batch, 64, frame_num=4, H=512, W=512]
        # print(torch.isnan(torch.sum(x)).item(), end=' ')
        x = self.pool1(x)
        # print(x.size())# [batch, 64, frame_num=4, H/2=256, W/2=256]
        # print(torch.isnan(torch.sum(x)).item(), end=' ')

        x = self.relu(self.conv2a(x))
        # print(x.size())# [batch, 128, frame_num=4, H/4=128, W/4=128]
        # print(torch.isnan(torch.sum(x)).item(), end=' ')
        x = self.relu(self.conv2b(x))
        # print(x.size())# [batch, 128, frame_num=4, H/8=64, W/8=64]
        # print(torch.isnan(torch.sum(x)).item(), end=' ')
        x = self.pool2(x)  
        # print(x.size())# [batch, 128, frame_num=4, H/16=32, W/16=32] 
        # print(torch.isnan(torch.sum(x)).item(), end=' ')

        x = self.relu(self.conv3(x))
        # print(x.size())# [batch, 256, frame_num=4, H/16=32, W/16=32]
        # print(torch.isnan(torch.sum(x)).item(), end=' ')
        x = self.pool3(x) 
        # print(x.size())# [batch, 256, frame_num=1, H/32=16, W/32=16] 
        # print(torch.isnan(torch.sum(x)).item(), end=' ')

        x = self.relu(self.conv4a(x))
        # print(x.size())# [batch, 512, frame_num=4, H/64=8, W/64=8] 
        # print(torch.isnan(torch.sum(x)).item(), end=' ')
        x = self.relu(self.conv4b(x)) 
        # print(x.size())# [batch, 512, frame_num=4, H/128=4, W/128=4]
        # print(torch.isnan(torch.sum(x)).item(), end=' ') 
        x = self.pool4(x) 
        # print(x.size())# [batch, 512, frame_num=4, H/256=2, W/256=2]
        # print(torch.isnan(torch.sum(x)).item(), end=' ')


        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # print(x.size())# [batch, f, 512, 2, 2]
        x = x.view(x.size(0), x.size(1),-1)
        # print(x.size())# [batch, f, 2048]
        x, hidden = self.gru(x, None)
        # print(x.size())# [batch, f, 512]
        x = self.pool(x)
        # print(x.size())# [batch, f, 256]
        x = self.linear0(x)
        # print(x.size())# [batch, f, 128]
        x = self.linear1(x)
        # print(x.size())# [batch, f, 64]
        x = self.linear2(x)
        # print(x.size())# [batch, f, num_classes=2]
        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

#VGG+GRU (224, 224) laye21개 중 conv 8개########################################################################################                
class vggGRU(nn.Module):
    def __init__(self, num_classes=2, input_type='rgb'):
        super().__init__()

        if input_type=='rgb': channel=3
        else: channel=2

        self.relu = nn.ReLU()

        self.m=nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.c1_1=nn.Conv3d(channel, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.c1_2=nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        self.c2_1=nn.Conv3d(64, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.c2_2=nn.Conv3d(128, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        self.c3_1=nn.Conv3d(128, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.c3_23=nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        self.c4_1=nn.Conv3d(256, 512, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.c4_=nn.Conv3d(512, 512, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        self.gru1=nn.GRU(input_size=512*7*7, hidden_size=4096, batch_first=True, bias=True, bidirectional=False)
        self.lin4hidden = nn.Linear(4096, 2048)
        self.gru2=nn.GRU(input_size=4096, hidden_size=2048, batch_first=True, bias=True, bidirectional=False)
        self.fc3=nn.Linear(2048, num_classes)

        self._initialize_weights()

    def forward(self, x):
        # print(x.size())#(batch, RGB, frame_len, H, W)
        x=self.relu(self.c1_1(x))
        # print(x.size())#(batch, RGB, frame_len, H, W)
        x=self.relu(self.c1_2(x))
        # print(x.size())#(batch, RGB, frame_len, H, W)
        x=self.m(x)
        # print(x.size())#(batch, RGB, frame_len, H/2, W/2)

        x=self.relu(self.c2_1(x))
        # print(x.size())#(batch, RGB, frame_len, H/2, W/2)
        x=self.relu(self.c2_2(x))
        # print(x.size())#(batch, RGB, frame_len, H/2, W/2)
        x=self.m(x)
        # print(x.size())#(batch, RGB, frame_len, H/4, W/4)

        x=self.relu(self.c3_1(x))
        # print(x.size())#(batch, RGB, frame_len, H/4, W/4)
        x=self.relu(self.c3_23(x))
        # print(x.size())#(batch, RGB, frame_len, H/4, W/4)
        x=self.relu(self.c3_23(x))
        # print(x.size())#(batch, RGB, frame_len, H/4, W/4)
        x=self.m(x)
        # print(x.size())#(batch, RGB, frame_len, H/8, W/8)

        x=self.relu(self.c4_1(x))
        # print(x.size())#(batch, RGB, frame_len, H/8, W/8)
        x=self.relu(self.c4_(x))
        # print(x.size())#(batch, RGB, frame_len, H/8, W/8)
        x=self.relu(self.c4_(x))
        # print(x.size())#(batch, RGB, frame_len, H/8, W/8)
        x=self.m(x)
        # print(x.size())#(batch, RGB, frame_len, H/16, W/16)

        x=self.relu(self.c4_(x))
        # print(x.size())#(batch, RGB, frame_len, H/16, W/16)
        x=self.relu(self.c4_(x))
        # print(x.size())#(batch, RGB, frame_len, H/16, W/16)
        x=self.relu(self.c4_(x))
        # print(x.size())#(batch, RGB, frame_len, H/16, W/16)
        x=self.m(x)
        # print(x.size())#(batch, RGB, frame_len, H/32, W/32)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # print(x.size())# [batch, f, RGB, 7, 7]
        x = x.view(x.size(0), x.size(1),-1)
        # print(x.size())#torch.Size([batch, f, 512*7*7 = 25088])
        x, hidden = self.gru1(x, None)
        # print(x.size())#torch.Size([batch, f, 4096])
        # print(hidden.size())#torch.Size([1, batch, 4096])
        hidden = self.lin4hidden(hidden)
        # print(hidden.size())#torch.Size([1, batch, 2048])
        x, hidden = self.gru2(x, hidden)
        # print(x.size())#torch.Size([batch, f, 2048])
        x=self.fc3(x)
        # print(x.size())#torch.Size([batch, f, 2])

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

#cnn3d+GRU+PonderNet (H=512, W=512)########################################################################################
class cnn3dPonder(nn.Module):#layer 15개 중 conv 6개
    def __init__(self, max_steps, allow_halting=False, num_classes=2, input_type='rgb'):
        super().__init__()

        if input_type=='rgb': channel=3
        else: channel=2

        self.preConv = nn.Sequential(nn.Conv3d(channel, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)), 
        nn.ReLU(), 
        nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)), 
        nn.Conv3d(64, 128, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2)), 
        nn.ReLU(), 
        nn.Conv3d(128, 128, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2)), 
        nn.ReLU(), 
        nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)), 
        nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)), 
        nn.ReLU(), 
        nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)), 
        nn.Conv3d(256, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2)), 
        nn.ReLU(), 
        nn.Conv3d(512, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2)), 
        nn.ReLU(), 
        nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))

        self.pool = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 1, 1))

        self.gru = nn.GRU(input_size=256, hidden_size=256, batch_first=True, bias=True, bidirectional=True)
        # self.gru = nn.GRUCell(input_size=512*2*2, hidden_size=256, bias=True)
        self.output_layer = nn.Sequential(nn.Linear(256, 128), nn.Linear(128, 64), nn.Linear(64, num_classes))
        self.lambda_layer = nn.Sequential(nn.Linear(256, 128), nn.Linear(128, 64), nn.Linear(64, num_classes))
        
        #outputs=torch.flatten(outputs, start_dim=0, end_dim=1)
        self.lambda_prob = nn.Softmax(dim=1)
        
        self.max_steps = max_steps
        self.is_halt = allow_halting
        self.num_classes = num_classes

        # self.__init_weight()

    def forward(self, x):

        x=self.preConv(x)
        # [batch, 512, frame_num=4, H/256=2, W/256=2]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # print(x.size())# [batch, f, 512, 2, 2]
        x = self.pool(x)
        # print(x.size())##[batch, f, 256, 1, 1]
        x = x.view(x.size(0), x.size(1),-1)
        # print(x.size())# [batch, f, 256]          
        x, hidden = self.gru(x, None)
        # print(x.size(), hidden.size())#torch.Size([batch, f, 256]) torch.Size([1, batch, 256])
        p=[]#halting probabilities. Sums over rows (fixing a sample)are 1.
        y=[]#predictions 
        batch_size = x.size(0)
        un_halted_prob = hidden.new_ones((batch_size, self.num_classes))
        #halting_step: An integer for each sample in the batch that corresponds to the step when it was halted. 
        halting_step = hidden.new_zeros((batch_size, self.num_classes), dtype=torch.long, device = x.device)
        for n in range(1, self.max_steps+1):
            if n == self.max_steps:
                lambda_n = hidden.new_ones((batch_size, self.num_classes))
            else:
                lambda_n = self.lambda_prob(self.lambda_layer(hidden))#torch.Size([1, 5, 2])
                lambda_n = lambda_n[0]#torch.Size([5, 2])
            
            # print(un_halted_prob.size(), lambda_n.size())#torch.Size([5, 2]) torch.Size([5, 2])
            p.append(un_halted_prob*lambda_n)
            y.append(self.output_layer(hidden)[0])#torch.Size([5, 2]) stack

            halting_step = torch.maximum(n * (halting_step == 0) * torch.bernoulli(lambda_n).to(torch.long), halting_step)
            un_halted_prob=un_halted_prob*(1-lambda_n)
            # print(x.size(), hidden.size())#torch.Size([batch, f, 256]) torch.Size([1, batch, 256])
            #input.size(-1) must be equal to input_size. Expected 2048, got 256
            x, hidden = self.gru(x, hidden)

            if self.is_halt and (halting_step > 0).sum()==batch_size:
                break

        return torch.stack(y), torch.stack(p), halting_step

    # def __init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d):
    #             # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             # m.weight.data.normal_(0, math.sqrt(2. / n))
    #             torch.nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, nn.BatchNorm3d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()

class ReconstructionLoss(nn.Module):
    def __init__(self, loss_func):
        super().__init__()
        self.loss_func=loss_func
    def forward(self, p, y_hat, y):
        total_loss = p.new_tensor(0.)
        for n in range(p.shape[0]):
            loss = (p[n]*self.loss_func(y_hat[n], y)).mean()
            print(loss)
            total_loss = total_loss+loss
        return total_loss

class RegularizationLoss(nn.Module):
    def __init__(self, lambda_p, max_steps):
        super().__init__()
        p_g=torch.zeros((max_steps,))
        not_halted=1.
        for k in range(max_steps):
            p_g[k]=not_halted*lambda_p
            not_halted=not_halted*(1-lambda_p)

        self.p_g=nn.Parameter(p_g, requires_grad=False)
        self.kl_div=nn.KLDivLoss(reduction='batchmean')

    def forward(self, p):
        p=p.transpose(0, 1)
        p_g=self.p_g[None, :p.shape[1]].expand_as(p)
        return self.kl_div(p.log(), p_g)

#penalty loss만들기################################################################
class PenaltyLoss(nn.Module):
    def __init__(self, loss_func):
        super().__init__()
        self.loss_func = loss_func
    def forward(self, outputs, label, preds):
        label_ind = label.nonzero()
        pred_ind = preds.nonzero()
        incorrect = 0
        if label.sum():
            SL = min(label_ind)
            EL = max(label_ind)
            if pred_ind.sum():
                for i in pred_ind:
                    if i < SL:
                        incorrect = incorrect + SL-i
                    elif i > EL:
                        incorrect = incorrect + i - EL
            else: 
                incorrect = max(SL, len(label)-EL)*len(label_ind)
        return self.loss_func(outputs, label)+incorrect


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class gruModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size = input_size, hidden_size = hidden_size, batch_first=True, bias=True, bidirectional=False, num_layers = num_layers)
        self.linear = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        x, hidden = self.gru(x, None)
        return self.linear(x)

class LinearModule(nn.Module):
    def __init__(self, input_len, frame_len, mlp_dim, num_classes):
        super().__init__()
        self.lin = nn.Linear(input_len, frame_len)
        self.classifier = nn.Linear(mlp_dim, num_classes)
    def forward(self, x): # x = (batch=input_len, mlp_dim)
        x = x.transpose(0, 1)# x = (mlp_dim, batch=input_len)
        x = self.lin(x)# x = (mlp_dim, frame_len)
        x = x.transpose(0, 1)# x = (frame_len, mlp_dim)
        x = self.classifier(x)# x = (frame_len, num_classes)
        return x

class Both1Lin(nn.Module):
    def __init__(self, forward_model, backward_model, cat_length, num_classes=2):
        super().__init__()
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.cat = nn.Linear(cat_length, num_classes)
    def forward(self, forward, backward, cat_dim=1):
        forward = self.forward_model(forward)
        backward = self.backward_model(backward)
        #[batch, features] 일 경우, cat_dim=1
        # print(forward.size())
        x = torch.cat((forward, backward), dim=cat_dim)
        if (len(x.size())==3) and (x.size(-1)==1): # if i3d model
            x = x.squeeze(2)
        # print(x.size())
        x = self.cat(x)
        # print(x.size())
        return x

class TriModel(nn.Module):
    def __init__(self, forward_model, central_model, backward_model, cat_length, num_classes=2):
        super().__init__()
        self.forward_model = forward_model
        self.central_model = central_model
        self.backward_model = backward_model
        self.cat = nn.Linear(cat_length, num_classes)
    def forward(self, forward, central, backward, cat_dim=1):
        forward = self.forward_model(forward)
        central = self.central_model(central)
        backward = self.backward_model(backward)
        x = torch.cat((forward, central, backward), dim=cat_dim)
        x = self.cat(x)
        return x


class Transformer_2D(nn.Module): # nhead 주로 1로 함. 다른거 많이 해봤지만 별다를건 없었나봄. vfss 학습시에 Adam optimizer를 사용한것만 실험했는데, SGD가더 잘나왔음
    def __init__(self, model, d_model, nhead=8, num_layers=6, num_classes=2):
        super().__init__()
        self.model=model
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2048, 
                                                        dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=None)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, src, mask=None):
        B, C, T, H, W = src.shape
        src = src.permute(0, 2, 1, 3, 4)
        src = src.reshape(B * T, C, H, W)
        src = self.model(src)
        src = src.reshape(B, T, -1)
        src = self.transformer(src, src_key_padding_mask = mask)
        src = self.fc(src)
        return src


class Transformer_3D(nn.Module): # optimizer SGD로 바꾸고 nhead도 다양하게 해보기. num_layers도 주로 1로만 했었는데, 복잡도 적게 1로 하는게 맞는거 같음...
    def __init__(self, d_model, dim_feedforward=2048, nhead=8, num_layers=6, num_classes=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
                                                        dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=True)
        '''
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))

        # self-attention block
        def _sa_block(self, x: Tensor,
                    attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
            x = self.self_attn(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=False)[0]
            return self.dropout1(x)

        # feed forward block
        def _ff_block(self, x: Tensor) -> Tensor:
            x = self.linear2(self.dropout(self.activation(self.linear1(x))))
            return self.dropout2(x)
        '''
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=None)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src, mask=None):
        # B, T, d_model = src.shape
        src = self.transformer(src, src_key_padding_mask = mask)
        src = self.fc(src)
        return src


class MHA_3D(nn.Module): # Multi Head Attention 3D
    def __init__(self, embed_dim, num_heads, num_classes):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, src, mask=None):
        src, _ = self.attn(src, src, src, key_padding_mask=mask) 
        src = self.fc(src)
        return src
    
class doubleConv1d(nn.Module): # adapted from the U-Net
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = doubleConv1d(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = doubleConv1d(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff = x2.size(-1) - x1.size(-1)
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1) # concatenate along the embed_dim dimension
        return self.conv(x)

class MHA_UNet(nn.Module): # Multi Head Attention & U-Net
    def __init__(self, embed_dim, num_heads, num_classes, bilinear=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.inc = doubleConv1d(in_channels=num_classes, out_channels=num_classes*2) # 채널 (feature) 늘리기
        self.down1 = nn.Sequential( 
            nn.MaxPool1d(2), # frameLength 2배 압축
            doubleConv1d(in_channels=num_classes*2, out_channels=num_classes*4) # 채널 (feature) 늘리기
        )
        factor = 2 if bilinear else 1
        self.up1 = Up(num_classes*4, num_classes*2 // factor, bilinear)
        self.outConv = nn.Conv1d(in_channels=num_classes*2, out_channels=num_classes, kernel_size=1)

    def forward(self, x, mask=None):
        x, _ = self.attn(x, x, x, key_padding_mask=mask) # [batch, frameLength, embed_dim]
        x = self.fc(x) # [batch, frameLength, num_classes]
        x = x.transpose(-1, -2) # [batch, num_classes, frameLength]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x = self.up1(x2, x1)
        x = self.outConv(x)
        x = x.transpose(-1, -2) # [batch, frameLength, num_classes]
        return x

class Transformer_UNet(nn.Module): # Multi Head Attention & U-Net
    def __init__(self, embed_dim, num_heads, num_classes, bilinear=False):
        super().__init__()
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*2, 
                                                        dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=True)
        # self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=embed_dim//2, nhead=num_heads, dim_feedforward=embed_dim//4, 
        #                                                 dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=True)
        self.inc = doubleConv1d(in_channels=embed_dim, out_channels=embed_dim*2) # 채널 (feature) 늘리기
        self.down1 = nn.Sequential( 
            nn.MaxPool1d(2), # frameLength 2배 압축
            doubleConv1d(in_channels=embed_dim*2, out_channels=embed_dim*4) # 채널 (feature) 늘리기
        )
        factor = 2 if bilinear else 1
        self.up1 = Up(embed_dim*4, embed_dim*2 // factor, bilinear)
        self.outConv = nn.Conv1d(in_channels=embed_dim*2, out_channels=num_classes, kernel_size=1)

    def forward(self, x, mask=None):
        # x, _ = self.attn(x, x, x) # [batch, frameLength, embed_dim]
        x = self.encoder_layer1(x, src_key_padding_mask = mask)
        # x = self.encoder_layer2(x)
        x = x.transpose(-1, -2) # [batch, embed_dim, frameLength]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x = self.up1(x2, x1)
        x = self.outConv(x)
        x = x.transpose(-1, -2) # [batch, frameLength, embed_dim]
        return x

class PE(nn.Module):
    "Implement the positional encoding function."

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        import math
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # [[1], [2], ..., [max_len]]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x, start_f=None):
        if start_f is not None:
            f = x.size(1)
            for b in range(x.size(0)):
                x[b] = x[b]+self.pe[:, start_f[b] : start_f[b] + f].requires_grad_(False)
        else:
            x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)      

class PeB4Fc(nn.Module): 
    def __init__(self, backbone, PE, FC, freeze=False):
        super(PeB4Fc, self).__init__()
        self.backbone = backbone
        self.PE = PE
        self.FC = FC
        self.freeze = freeze # whether freeze backbone module
    
    def forward(self, x, start_f=None, mask=None):
        if self.freeze:
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)
        x = self.PE(x, start_f)
        x = self.FC(x, mask)
        return x

class LSTM(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LSTM, self).__init__()    
        #self.h0 = torch.zeros(1, )
        self.num_layers=1
        self.feature_dim = feature_dim
        self.lstm = torch.nn.LSTM(input_size = feature_dim, hidden_size = feature_dim, batch_first=True, bias=True, bidirectional=True, num_layers = 1)
        # self.lin1 = torch.nn.Linear(2*feature_dim, feature_dim) 
        # self.lin2 = torch.nn.Linear(feature_dim, num_classes) 
        self.lin = torch.nn.Linear(2*feature_dim, num_classes)

    def forward(self, x):
        h0=torch.zeros(2*self.num_layers, x.size(0), self.feature_dim, device=x.device, requires_grad=True)
        c0=torch.zeros(2*self.num_layers, x.size(0), self.feature_dim, device=x.device, requires_grad=True)
        x, (h0, c0)=self.lstm(x, (h0, c0))#[batch, frame_len, feature_dim]
        # x=self.lin(x[:, -1])#[batch, num_classes]
        x=self.lin(x)
        # x=self.lin1(x)
        # x=self.lin2(x)
        return x
        

# from cross_attention import CrossAttentionAdapterLayer as ca_adapter
# class adapter(nn.Module):
#     def __init__(self, rgb_model, flow_model, d_model, pf_dim):
#         super(adapter, self).__init__() 
#         self.rgb_model=rgb_model
#         self.flow_model=flow_model
#         self.teacher = ca_adapter(hid_dim=d_model, n_heads=1, pf_dim=pf_dim, dropout=0.1)
#         self.student = 
#     def forward(self, rgb, flow):
#         rgb=self.rgb_model(rgb)
#         flow=self.flow_model(flow)
#         output = self.teacher(rgb, flow)
#         return output

# https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
class AsymmetricLoss(nn.Module): 
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0, eps=0, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        # 원래 값: gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
 
    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
 
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
 
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
 
        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
 
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
 
        return -loss.sum()


# # https://github.com/mlyg/unified-focal-loss/blob/411c9c5ce43b2ef847f0903d1b841512ad8d2eee/loss_functions.py#L260
# class AsymmetricFocalLoss(nn.Module):
#     def __init__(self, delta=0.7, gamma=2., epsilon=1e-7):
#         super().__init__()
#         """For Imbalanced datasets
#         Parameters
#         ----------
#         delta : float, optional
#             controls weight given to false positive and false negatives, by default 0.7
#         gamma : float, optional
#             Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
#         """
#         self.delta=delta
#         self.gamma=gamma
#         self.epsilon=epsilon
#     def forward(self, y_true, y_pred):
#         y_pred=torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
#         cross_entropy = -y_true * torch.log(y_pred) 
#         #calculate losses separately for each class, only suppressing background class
#         # 이 줄 이하로 shape 확인하기!!!!!!
#         # print('shape 확인', cross_entropy.shape, y_pred.shape) # torch.Size([4, 7, 16]) torch.Size([4, 7, 16])
#         # shape (batch, W, H, binary mask). binary를 위한 것. ㅠㅠ
#         back_ce = torch.pow(1 - y_pred[:,:,:,0], self.gamma) * cross_entropy[:,:,:,0] 
#         back_ce =  (1 - self.delta) * back_ce

#         fore_ce = cross_entropy[:,:,:,1]
#         fore_ce = self.delta * fore_ce

#         loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce],dim=-1),dim=-1))

#         return loss

# https://github.com/abhuse/polyloss-pytorch/blob/main/polyloss.py
import torch.nn.functional as F
from torch import Tensor
class Poly1FocalLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "none",
                 weight: Tensor = None,
                 pos_weight: Tensor = None,
                 label_is_onehot: bool = True):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to binary Cross-Entropy loss
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param labels: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise 
            one-hot encoded tensor of same shape as logits
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py

        p = torch.sigmoid(logits)

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=self.num_classes)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                labels = F.one_hot(labels.unsqueeze(1), self.num_classes).transpose(1, -1).squeeze_(-1)

        labels = labels.to(device=logits.device,
                           dtype=logits.dtype)

        ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels,
                                                     reduction="none",
                                                     weight=self.weight,
                                                     pos_weight=self.pos_weight)
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()

        return poly1
