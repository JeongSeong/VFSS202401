import torch
import os
from PIL import Image
import numpy as np
import random
import pandas as pd
import copy
from tqdm import tqdm
import multiprocessing

from vfssData import video_to_tensor, calIoU, make_label_array
import MultiFrameLoadUtil


def load_rgb_frames(image_dir, vid, f_ind, transforms, pool, resize=None, gray_scale=False):#resize=(H, W)
    
    #   frames = []
    P_dir = os.path.join(image_dir, str(vid).zfill(4))
    fnameList = np.array(list(pool.map(MultiFrameLoadUtil.addJpg, sorted(list(pool.map(MultiFrameLoadUtil.extractName, os.listdir(P_dir)))))))[f_ind]

    mode = 'L' if gray_scale else 'RGB'
    imageFn = MultiFrameLoadUtil.imageTransform(P_dir=P_dir, mode=mode, size=resize, resample=Image.LANCZOS)
    fnameList = pool.map(imageFn.imageOpen, fnameList)
    #   print(fnameList)
    if resize is not None: 
        fnameList = pool.map(imageFn.resizeImage, fnameList)
    #   print(fnameList)
    fnameList = list(pool.map(MultiFrameLoadUtil.makeArray, fnameList))
    #   print(fnameList)
    if transforms is not None:#when modelName == 'vgg'
        # normalize image between 0 and 1 # img = (img-np.min(img))/(np.max(img)-np.min(img))
        fnameList = np.array(pool.map(MultiFrameLoadUtil.normBTW_zeroN_one, fnameList), dtype=np.float32)#(T x H x W x C)
        fnameList = torch.from_numpy(fnameList.transpose([0,3,1,2]))#(T x C x H x W)
        fnameList = transforms(fnameList)
        fnameList = fnameList.transpose(0,1)
    else:# normalize image between -1 and 1 #np.max(img)==255 #
        fnameList = np.array(pool.map(MultiFrameLoadUtil.normBTW_absOne, fnameList), np.float32)#(T x H x W x C)
        fnameList = torch.from_numpy(fnameList.transpose([3,0,1,2]))
    #   print(fnameList.shape)
    #Size([batch, RGB, frame_len, H, W])
    return fnameList 

def load_flow_frames(image_dir, vid, f_ind, transforms, pool, resize=None):
    P_dir = os.path.join(image_dir, str(vid).zfill(4))
    fnameList = np.array(sorted(set(list(pool.map(MultiFrameLoadUtil.extractPID, os.listdir(P_dir))))))[f_ind[:-1]]
    fnameListX = np.array(list(pool.map(MultiFrameLoadUtil.addXJpg, fnameList)))
    fnameListY = np.array(list(pool.map(MultiFrameLoadUtil.addYJpg, fnameList)))
    imageFn = MultiFrameLoadUtil.imageTransform(P_dir=P_dir, mode='L', size=resize, resample=Image.LANCZOS)
    fnameListX = pool.map(imageFn.imageOpen, fnameListX)
    fnameListY = pool.map(imageFn.imageOpen, fnameListY)
    if resize is not None: 
        fnameListX = pool.map(imageFn.resizeImage, fnameListX)
        fnameListY = pool.map(imageFn.resizeImage, fnameListY)
    fnameListX = list(pool.map(MultiFrameLoadUtil.makeArray, fnameListX))
    fnameListY = list(pool.map(MultiFrameLoadUtil.makeArray, fnameListY))
    if transforms is not None:#when modelName == 'vgg' but in flow,,, transforms does not work.
        # normalize image between 0 and 1 # img = (img-np.min(img))/(np.max(img)-np.min(img))
        fnameListX = np.array(pool.map(MultiFrameLoadUtil.normBTW_zeroN_one, fnameListX), dtype=np.float32)#(T x H x W)
        fnameListY = np.array(pool.map(MultiFrameLoadUtil.normBTW_zeroN_one, fnameListY), dtype=np.float32)#(T x H x W)
    else:# normalize image between -1 and 1 #np.max(img)==255 #
        fnameListX = np.array(pool.map(MultiFrameLoadUtil.normBTW_absOne, fnameListX), np.float32)#(T x H x W)
        fnameListY = np.array(pool.map(MultiFrameLoadUtil.normBTW_absOne, fnameListY), np.float32)#(T x H x W)
    # print('fnameListX.shape', fnameListX.shape, ', fnameListY.shape', fnameListY.shape)
    fnameList = torch.stack((torch.from_numpy(fnameListX), torch.from_numpy(fnameListY))) # (C, T, H, W)

    return fnameList #Size([batch, xy, frame_len, H, W])

def make_dataset_withFrames(split_file, split, rgb_root, flow_root, resize, gray_scale, transforms): # default: rgb_root=None, flow_root=None
    dataset = []
    pool = multiprocessing.Pool()
    df = pd.read_excel(split_file)
    
    idx= df.loc[df['split']==split].index
    #num_classes=2로 설계돼있음
    for i in tqdm(idx):
        vid = df.loc[i, 'patient_id']
        # print('patentID', vid)
        if rgb_root is not None and not os.path.exists(os.path.join(rgb_root, str(vid).zfill(4))):
            print(f'no rgb frames for {str(vid).zfill(4)}!')
            continue
        if flow_root is not None and not os.path.exists(os.path.join(flow_root, str(vid).zfill(4))):
            print(f'no flow frames for {str(vid).zfill(4)}!')
            continue

        lvc_start = min(df.loc[i, 'label_start'], df.loc[i, 'label_end'])
        lvc_end = max(df.loc[i, 'label_start'], df.loc[i, 'label_end']) 
        num_frame = df.loc[i, 'num_frame']
        f_ind = list(range(0, num_frame))

        if rgb_root is not None:
            rgb = load_rgb_frames(rgb_root, vid, f_ind, transforms, pool, resize, gray_scale)
        else:
            rgb = None

        if flow_root is not None:# self.input_type = 'both'
            flow = load_flow_frames(flow_root, vid, f_ind, transforms, pool, resize)
        else:
            flow=None

        dataset.append((vid, num_frame, lvc_start, lvc_end, rgb, flow))
    pool.close()
    return dataset



class VFSS_data(torch.utils.data.Dataset): # root, input_type 사라지고 rgb_root, flow_root 생김, fill 추가됨

    def __init__(self, split_file, split, frame_len, sampling, transforms, gray_scale=False, fill=None
    , rgb_root=None, flow_root=None, num_classes=2, resize=None, load_whole= False, labelType = None):

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
        if split == 'test':
            import sys
            sys.exit('not supported yet')
            from vfssData_withSkippedOrInterpolated import make_dataset
            self.data = make_dataset(split_file, split, rgb_root, flow_root)
        else:
            self.data = make_dataset_withFrames(split_file, split, rgb_root, flow_root, resize, gray_scale, transforms) 


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
        vid, num_frame, SL, EL, rgb, flow = self.data[index]#, label

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
        
        elif 'PE' in self.sampling:
            if ':' in self.sampling: # 'PE3:1' 형식. N:1만 가능
                ratio = self.sampling.replace('PE', '')
                ratio = ratio.split(':')

                if self.split == 'train':
                    coin = random.randint(0,int(ratio[0])-1)
                    if coin==0:
                        start_f = random.randint(max(0, SL-self.frame_len), min(EL, num_frame-6))
                    else:# f_ind 하나만 뽑히는 경우에 flows 가져올 때 에러남, test 시 overlap 해서 앞의 정보를 어느정도 주는 게 좋음
                        start_f = random.randint(0, num_frame-6)
                else:# 앞의 6개 프레임의 정보를 주는 이유는, I3D나 bidirectional 할 때, 13 프레임씩 넣어, 가운데 프레임을 예측했었는데
                    start_f = random.randint(0, num_frame-6)
            else:# 그럼 그 가운데 프레임은 앞의 6개 프레임을 보는 것이므로, 그것과 유사하게 test 하겠다.
                start_f = random.randint(0, num_frame-6)

            f_ind=list(range(start_f, min(start_f+self.frame_len, num_frame)))
                       
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
                # revised sampling 별로임. 이 경우는 사용 안함.
                padding = list(label.shape)
                padding[1] = -start_f
                padding = np.zeros(padding, np.float32)
                padding[0, :] = 1
                label = np.concatenate((padding, label), axis=1)
            if start_f > (num_frame-self.frame_len):
                padding = list(label.shape)
                padding[1] = start_f+self.frame_len - num_frame
                padding = np.zeros(padding, np.float32)
                if self.sampling == 'PECE': # criterion CE 일때만 해당 <--- 얘 하는것보다 nested tensor랑 collate function 쓰는 게 더 나을 듯.
                    padding[1, :] = -1
                else:
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
        time_dim=1
        if self.rgb_root is not None:
            rgb = torch.index_select(input=rgb, dim=1, index=torch.tensor(f_ind))
            if (self.frame_len == 1):
                rgb = rgb.squeeze(time_dim)
            elif self.frame_len!=1:
                if start_f < 0:
                    size = list(rgb.size())
                    size[time_dim] = -start_f
                    rgb = torch.cat((rgb.new_zeros(size), rgb), dim=time_dim)
                if start_f > (num_frame-self.frame_len):
                    size = list(rgb.size())
                    size[time_dim] = start_f+self.frame_len - num_frame
                    rgb = torch.cat((rgb, rgb.new_zeros(size)), dim=time_dim)
                # if rgb.size(1) != self.frame_len:
                #     print(rgb.size(), start_f, num_frame, num_frame-self.frame_len)

        if self.flow_root is not None:
            flow = torch.index_select(input=flow, dim=1, index=torch.tensor(f_ind[:-1]))
            if self.frame_len == 1:
                flow = flow.squeeze(time_dim)
            elif self.frame_len!=1:
                if start_f < 0: # this must be revised sampling, frame_len > 1
                    size = list(flow.size())
                    size[time_dim] = -start_f
                    flow = torch.cat((flow.new_zeros(size), flow), dim=time_dim)
                if start_f > (num_frame-self.frame_len):
                    size = list(flow.size())
                    size[time_dim] = start_f+self.frame_len - num_frame
                    flow = torch.cat((flow, flow.new_zeros(size)), dim=time_dim)

        if self.rgb_root is not None:
            if self.flow_root is not None:
                '''rgb, flow 둘 다 넣었을 때 반환값'''
                if 'PE' in self.sampling:
                    return rgb, flow, torch.from_numpy(label), vid, torch.tensor(start_f)
                else:
                    return rgb, flow, torch.from_numpy(label), iouLabel, vid    
            else: # self.input_type = 'rgb'
                if 'PE' in self.sampling:
                    return rgb, torch.from_numpy(label), vid, torch.tensor(start_f)
                else:
                    return rgb, torch.from_numpy(label), iouLabel, vid
        else: #if self.flow_root is not None: # self.input_type = 'flow'
            if 'PE' in self.sampling:
                return flow, torch.from_numpy(label), vid, torch.tensor(start_f)
            else:
                return flow, torch.from_numpy(label), iouLabel, vid


    def __len__(self):
         return len(self.data)
