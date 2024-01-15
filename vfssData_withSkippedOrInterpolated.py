import torch
import os
from PIL import Image
import numpy as np
import random
import pandas as pd
import copy
from vfssData import video_to_tensor, calIoU, make_label_array


def load_rgb_frames(image_dir, vid, f_ind, resize=None, gray_scale=False):#resize=(H, W)
    #   frames = []
    P_dir = os.path.join(image_dir, str(vid).zfill(4))
    #   print(P_dir)
    #   print(list(map(lambda x: x+'.jpg', sorted(list(map(lambda x: x.split('.')[0], os.listdir(P_dir)))))))
    fnameList = np.array(list(map(lambda x: x+'.jpg', sorted(list(map(lambda x: x.split('.')[0], os.listdir(P_dir)))))))[f_ind]
    #   print(fnameList)
    mode = 'L' if gray_scale else 'RGB'
    fnameList = map(lambda x: Image.open(os.path.join(P_dir, x)).convert(mode), fnameList)
    #   print(fnameList)
    if resize is not None: 
        fnameList = map(lambda x: x.resize(size=resize, resample=Image.LANCZOS), fnameList)
    #   print(fnameList)
    fnameList = list(map(lambda x: np.array(x), fnameList))
    #   print(fnameList)
    if len(f_ind)==1:#when modelName == 'vgg'
        # normalize image between 0 and 1 # img = (img-np.min(img))/(np.max(img)-np.min(img))
        fnameList = map(lambda img: (img-np.min(img))/(np.max(img)-np.min(img)), fnameList)
    else:# normalize image between -1 and 1 #np.max(img)==255 #
        fnameList = map(lambda img: (img/np.max(img))*2 - 1, fnameList)
    #   print(list(fnameList))
    fnameList = np.array(list(fnameList), np.float32) #fnameList = np.fromiter(fnameList, np.float32)
    #   print(fnameList.shape)
    return fnameList 

def load_flow_frames(image_dir, vid, f_ind, resize=None):
    P_dir = os.path.join(image_dir, str(vid).zfill(4))
    frames = np.array(sorted(set(list(map(lambda x: x.split('.')[0][:-1], os.listdir(P_dir))))))[f_ind[:-1]]
    for index, f in enumerate(frames):
        fx = os.path.join(P_dir, f+'x.jpg')
        fy = os.path.join(P_dir, f+'y.jpg')
        
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
            if index == 0:
                img = [np.asarray([imgx, imgy])] # (channel=2, H, W)
            else:
                img.append(np.asarray([imgx, imgy])) # (frame_len, channel=2, H, W)
        else:
            print('no flow!')    
    return np.asarray(img, dtype=np.float32).transpose([1, 0, 2, 3])

def make_dataset(split_file, split, rgb_root, flow_root): # default: rgb_root=None, flow_root=None
    dataset = []

    df = pd.read_excel(split_file)
    
    idx= df.loc[df['split']==split].index
    #num_classes=2로 설계돼있음
    for i in idx:
      num_frame = df.loc[i, 'num_frame']
      lvc_start = min(df.loc[i, 'label_start'], df.loc[i, 'label_end'])
      lvc_end = max(df.loc[i, 'label_start'], df.loc[i, 'label_end']) 

      vid = df.loc[i, 'patient_id']
      
      if rgb_root is not None and not os.path.exists(os.path.join(rgb_root, str(vid).zfill(4))):
        print(f'no rgb frames for {str(vid).zfill(4)}!')
        continue
      if flow_root is not None and not os.path.exists(os.path.join(flow_root, str(vid).zfill(4))):
        print(f'no flow frames for {str(vid).zfill(4)}!')
        continue

      dataset.append((vid, num_frame, lvc_start, lvc_end))
    
    return dataset



class VFSS_data(torch.utils.data.Dataset): # root, input_type 사라지고 rgb_root, flow_root 생김, fill 추가됨

    def __init__(self, split_file, split, frame_len, sampling, transforms, gray_scale=False, fill=None
    , rgb_root=None, flow_root=None, num_classes=2, resize=None, load_whole= False):

        # if randomSeed is not None: random.seed(randomSeed)
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
        vid, num_frame, SL, EL= self.data[index]#, label

        if self.load_whole:
          f_ind = list(range(0, num_frame))

        elif self.frame_len==1: #make positive sample one third!!
            coin = random.randint(0,2)
            if (self.sampling == 'revised2') and (self.split == 'val'): condition = (coin==0) or (coin==1)
            else: condition = (coin==2)

            if condition:
                positive = list(range(SL, EL+1))
                f_ind = [random.choice(positive)]
            else:
                negative = list(range(0, SL))+list(range(EL+1, num_frame))
                f_ind = [random.choice(negative)]
            # print('\n', vid, num_frame, SL, EL, f_ind)

        elif self.sampling[0] == 'p':
            flen = self.frame_len
            if self.split == 'train':
                coin = random.randint(0,9)
                cth = 3
                if coin < cth: #if it should be a positive label
                    offset = random.randint(0,8)
                    start_f = min(max(SL - 0 - offset,1), num_frame-flen)
                else:#if it should be a background label
                    start_f = random.randint(1,num_frame-flen)   
            else:
                start_f = random.randint(1,num_frame-flen)

            f_ind=list(range(0+start_f, 0+start_f+flen))
        
        elif self.sampling == 'revised':
            if self.split == 'train':
                coin = random.randint(0,2)
                if coin==2:
                    start_f = random.randint(SL-(self.frame_len//2), EL-(self.frame_len//2))
                    # 음수가 나오면 그 수만큼 0으로 padding
                    # frames.new_zeros(())
                else:
                    start_f = random.randint(-(self.frame_len//2), num_frame-(self.frame_len//2))
            else: # if self.split == 'val'
                start_f = random.randint(-(self.frame_len//2), num_frame-(self.frame_len//2))

            f_ind=list(range(max(0, start_f), min(start_f+self.frame_len, num_frame)))

        elif self.sampling == 'revised2': # 성능 완전 별로임
            if self.split == 'train':
                coin = random.randint(0,2)
                if coin==2: # 1/3 for positive
                    start_f = random.randint(SL-(self.frame_len//2), EL-(self.frame_len//2))
                    # 음수가 나오면 그 수만큼 0으로 padding
                    # frames.new_zeros(())
                else: # 2/3 for negative
                    start_f = random.randint(-(self.frame_len//2), num_frame-(self.frame_len//2))
            else: # if self.split == 'val'
                coin = random.randint(0,2)
                if coin==2: # 1/3 for negative
                    start_f = random.randint(-(self.frame_len//2), num_frame-(self.frame_len//2))
                else: # 2/3 for positive
                    start_f = random.randint(SL-(self.frame_len//2), EL-(self.frame_len//2))
            
            f_ind=list(range(max(0, start_f), min(start_f+self.frame_len, num_frame)))                
        
        IoU=calIoU(set(f_ind), SL, EL)    
        IoU_TH = self.frame_len / len(range(0, num_frame))
        iouLabel = 1 if IoU >= IoU_TH else 0

        if self.load_whole: 
            '''모든 프레임 추출 시 (test 시) 반환값'''
            return torch.tensor(f_ind, dtype=torch.int32), SL, EL, iouLabel, vid
            
        label = make_label_array(f_ind, SL, EL, self.num_classes)
        if self.frame_len!=1:
            if start_f < 0: # this must be revised sampling, frame_len > 1
                # np.zeros((num_classes, len(f_ind)), np.float32)
                padding = list(label.shape)
                padding[1] = -start_f
                padding = np.zeros(padding, np.float32)
                padding[0, :] = 1
                label = np.concatenate((padding, label), axis=1)
            elif start_f > (num_frame-self.frame_len):
                padding = list(label.shape)
                padding[1] = start_f+self.frame_len - num_frame
                padding = np.zeros(padding, np.float32)
                padding[0, :] = 1
                label = np.concatenate((label, padding), axis=1)

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
            rgb = load_rgb_frames(self.rgb_root, vid, f_ind, self.resize, self.gray_scale)
            rgb = video_to_tensor(rgb) #(C x T x H x W)
            if self.transforms is not None: # when vgg (frame_len=1) 
                rgb = rgb.squeeze(1)
                rgb = self.transforms(rgb)
            if self.frame_len!=1:
                if start_f < 0:
                    size = list(rgb.size())
                    size[1] = -start_f
                    rgb = torch.cat((rgb.new_zeros(size), rgb), dim=1)
                elif start_f > (num_frame-self.frame_len):
                    size = list(rgb.size())
                    size[1] = start_f+self.frame_len - num_frame
                    rgb = torch.cat((rgb, rgb.new_zeros(size)), dim=1)
                # if rgb.size(1) != self.frame_len:
                #     print(rgb.size(), start_f, num_frame, num_frame-self.frame_len)

        if self.flow_root is not None:# self.input_type = 'both'
            flow = load_flow_frames(self.flow_root, vid, f_ind, self.resize)
            flow = video_to_tensor(flow)
            if self.transforms is not None:
                flow = flow.squeeze(1)
                flow = self.transforms(flow)
            if self.frame_len!=1:
                if start_f < 0: # this must be revised sampling, frame_len > 1
                    size = list(flow.size())
                    size[1] = -start_f
                    flow = torch.cat((flow.new_zeros(size), flow), dim=1)
                elif start_f > (num_frame-self.frame_len):
                    size = list(flow.size())
                    size[1] = start_f+self.frame_len - num_frame
                    flow = torch.cat((flow, flow.new_zeros(size)), dim=1)

        if self.rgb_root is not None:
            if self.flow_root is not None:
                '''rgb, flow 둘 다 넣었을 때 반환값'''
                return rgb, flow, torch.from_numpy(label), iouLabel, vid    
            else: # self.input_type = 'rgb'
                return rgb, torch.from_numpy(label), iouLabel, vid
        else: #if self.flow_root is not None: # self.input_type = 'flow'
            return flow, torch.from_numpy(label), iouLabel, vid


    def __len__(self):
         return len(self.data)
