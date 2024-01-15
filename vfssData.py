import torch
import os
from PIL import Image
import numpy as np
import random
import pandas as pd
import copy

def video_to_tensor(pic):
    """
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))

def load_rgb_frames(image_dir, vid, f_ind, resize=None, gray_scale=False):#resize=(H, W)
  frames = []

  for i in f_ind:

    fname = os.path.join(image_dir,
                         str(vid).zfill(4), str(vid).zfill(4)+'_'+str(i).zfill(5)+'.jpg')

    if gray_scale:
        img = Image.open(fname).convert('L')
    else:
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
        img = np.asarray([imgx, imgy]) # (channel=2, H, W)
        frames.append(img) # (frame_len, channel=2, H, W)
    else:
        print('no flow!')    
  return np.asarray(frames, dtype=np.float32).transpose([1, 0, 2, 3])

def make_label_array(f_ind, SL, EL, num_classes):
    label = np.zeros((num_classes, len(f_ind)), np.float32)
    #   label[1, lvc_start-start_frame:lvc_end-start_frame+1] = 1
    #   label[0, :] = 1 - label[1, :]
    for i in range(len(f_ind)):
        if SL <= f_ind[i] <= EL:
            label[1, i] = 1
        else:
            label[0, i] = 1
    return label

def make_dataset(split_file, split, rgb_root, flow_root): # default: rgb_root=None, flow_root=None
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
      
      if rgb_root is not None and not os.path.exists(os.path.join(rgb_root, str(vid).zfill(4))):
        print(f'no rgb frames for {str(vid).zfill(4)}!')
        continue
      if flow_root is not None and not os.path.exists(os.path.join(flow_root, str(vid).zfill(4))):
        print(f'no flow frames for {str(vid).zfill(4)}!')
        continue

      dataset.append((vid, start_frame, end_frame, num_frame, lvc_start, lvc_end))
    
    return dataset

def calIoU(f_ind, SL, EL):# how much of GT is contained? the input must not have duplicates
    sum=0
    for i in f_ind:
        if i in range(SL, EL+1):
            sum+=1
    return sum/(EL+1-SL)

class VFSS_data(torch.utils.data.Dataset): # root, input_type 사라지고 rgb_root, flow_root 생김, fill 추가됨

    def __init__(self, split_file, split, frame_len, sampling, transforms, randomSeed=None, gray_scale=False, fill=None
    , rgb_root=None, flow_root=None, num_classes=2, resize=None, load_whole= False):

        if randomSeed is not None: random.seed(randomSeed)
        self.split_file = split_file
        self.transforms = transforms
        self.rgb_root = rgb_root
        self.flow_root = flow_root
        self.frame_len = frame_len
        self.fill = fill # fill the length with iteration
        self.load_whole = load_whole
        self.resize = resize
        self.sampling = sampling
        self.split = split
        self.num_classes = num_classes
        self.gray_scale = gray_scale
        self.data = make_dataset(split_file, split, rgb_root, flow_root) 

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            모든 프레임 추출 시 (test 시) 반환값
                torch.tensor(f_ind, dtype=torch.int32), SL, EL, iouLabel, vid
            rgb, flow 둘 다 넣었을 때 반환값
                rgb, flow, torch.from_numpy(label), iouLabel, vid
            rgb 혹은 flow만 넣었을 때 반환값
                imgs, torch.from_numpy(label), iouLabel, vid

            예전엔 iouLabel 대신 IoU를 반환했었음, 그런데 이젠 둘다 안씀
        """
        vid, start_frame, end_frame, num_frame, SL, EL= self.data[index]#, label

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
        
        IoU=calIoU(set(f_ind), SL, EL)    
        IoU_TH = self.frame_len / len(range(start_frame, end_frame+1))
        iouLabel = 1 if IoU >= IoU_TH else 0

        if self.load_whole: 
            '''모든 프레임 추출 시 (test 시) 반환값'''
            return torch.tensor(f_ind, dtype=torch.int32), SL, EL, iouLabel, vid
            
        label = make_label_array(f_ind, SL, EL, self.num_classes)

        if self.fill is not None:
            filler = self.fill - len(f_ind)
            # print('\n', filler, self.fill, len(f_ind))
            if filler != 0:
                new_f_ind = copy.deepcopy(f_ind)
                # print(len(new_f_ind))
                while len(new_f_ind) < filler:
                    new_f_ind.extend(f_ind)
                # print(len(new_f_ind), len(f_ind))
                new_f_ind = new_f_ind[:filler]
                # print(len(new_f_ind))
                # print()
                # print('f_ind', f_ind, type(f_ind), len(f_ind))
                # print('new_f_ind', new_f_ind, type(new_f_ind), len(new_f_ind))
                f_ind = f_ind+new_f_ind
                # print(f_ind, len(f_ind))

        if self.rgb_root is not None:
            if self.flow_root is not None:# self.input_type = 'both'
                rgb = load_rgb_frames(self.rgb_root, vid, f_ind, self.resize, self.gray_scale)
                rgb = video_to_tensor(rgb)
                if self.transforms is not None:
                    rgb = rgb.squeeze(1)
                    rgb = self.transforms(rgb)
                flow = load_flow_frames(self.flow_root, vid, f_ind, self.resize)
                flow = video_to_tensor(flow)
                if self.transforms is not None:
                    flow = flow.squeeze(1)
                    flow = self.transforms(flow)
                '''rgb, flow 둘 다 넣었을 때 반환값'''
                return rgb, flow, torch.from_numpy(label), iouLabel, vid    

            else: # self.input_type = 'rgb'
                imgs = load_rgb_frames(self.rgb_root, vid, f_ind, self.resize, self.gray_scale)
        elif self.flow_root is not None: # self.input_type = 'flow'
            imgs = load_flow_frames(self.flow_root, vid, f_ind, self.resize)
        if self.gray_scale: imgs=np.expand_dims(imgs, axis=-1)
        imgs = video_to_tensor(imgs)#(C x T x H x W)
        if self.transforms is not None:
            imgs = imgs.transpose(0,1)#(T x C x H x W)
            imgs = self.transforms(imgs)
            imgs = imgs.transpose(0,1)#(C x T x H x W)
        '''rgb 혹은 flow만 넣었을 때 반환값'''
        return imgs, torch.from_numpy(label), iouLabel, vid

    def __len__(self):
         return len(self.data)
