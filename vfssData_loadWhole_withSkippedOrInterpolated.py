# _PE1stF, 업데이트 됨
# test (load whole)를 위한 _multiLabel 업데이트 됨 
import torch
# import os
# from PIL import Image
import numpy as np
# import random
# import pandas as pd
from skimage.util import view_as_windows
from vfssData_withSkippedOrInterpolatedFixed import load_rgb_frames, load_flow_frames
from vfssData_multiLabel import make_event_label_array
from vfssData import calIoU, make_label_array

class VFSS_data(torch.utils.data.Dataset): # root, input_type 사라지고 rgb_root, flow_root 생김, fill 추가됨

    def __init__(self, split_file, split, frame_len, sampling, transforms, win_stride, gray_scale=False, fill=None
    , rgb_root=None, flow_root=None, num_classes=2, resize=None, front_rear_pad=0): #, load_whole= False):

        self.split_file = split_file
        self.transforms = transforms
        self.rgb_root = rgb_root
        self.flow_root = flow_root
        
        self.fill = fill # fill the length with iteration
        self.resize = resize
        self.sampling = sampling
        self.split = split
        self.num_classes = num_classes
        self.gray_scale = gray_scale
        self.win_stride = win_stride
        self.front_rear_pad = front_rear_pad

        if self.win_stride > 1:
            if self.rgb_root is not None: self.rgb_padding = None 
            if self.flow_root is not None: self.flow_padding = None
            self.ind_padding = None
        # 밖에서 frame_len4dataset 으로 변수넘겨줬으면, bi인지 아닌지 알 필요 없다
        # if bi: self.frame_len = frame_len*2-1
        # else: self.frame_len = frame_len
        self.frame_len = frame_len
        if 'multi' in split_file.lower(): self.multi=True
        else: self.multi=False

        if split == 'test':
            from vfssData_withSkippedOrInterpolatedFixed import make_dataset # multiclass supported!
            self.data = make_dataset(split_file, split, rgb_root, flow_root)
        else:
            from vfssData_withSkippedOrInterpolatedPreLoaded import make_dataset_withFrames
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
        
        if self.split == 'test':
            if self.multi:
                vid, num_frame, bpm,  hyoid,  lvc,  uesOpen,  uesClose,  lvcOff = self.data[index]
            else:
                vid, num_frame, SL, EL = self.data[index]
            f_ind = list(range(0, num_frame))
            if self.rgb_root is not None:
                rgb = load_rgb_frames(self.rgb_root, vid, f_ind, self.transforms, self.resize, self.gray_scale)
            else: rgb=None
            if self.flow_root is not None:
                flow = load_flow_frames(self.flow_root, vid, f_ind, self.transforms, self.resize)
            else: flow=None
        else:
            if self.multi: 
                import sys
                sys.exit('multi class data loader is only for test in this python file')
            else:
                vid, num_frame, SL, EL, rgb, flow = self.data[index]#, label
                f_ind = list(range(0, num_frame))
        # print(vid, rgb.shape, flow.shape)
        time_dim=0
        if self.rgb_root is not None:
            rgb=rgb.transpose(0,1)#(C x T x H x W)->(T x C x H x W)
            rgb=rgb.numpy()
            if self.gray_scale: rgb=np.expand_dims(rgb, axis=-1)
            size = list(rgb.shape)
            if self.front_rear_pad != 0:
                size[time_dim] = self.front_rear_pad
                padding = np.zeros(size, dtype=np.float32)
                rgb = np.concatenate((padding, rgb, padding))
            elif (self.win_stride > 1): # and (self.win_stride == self.frame_len) 
                self.rgb_padding = self.frame_len - size[time_dim] % self.win_stride
                # print(self.frame_len, size[time_dim], self.win_stride, size[time_dim] % self.win_stride, self.rgb_padding)
                size[time_dim] = self.rgb_padding
                padding = np.zeros(size, dtype=np.float32)
                rgb = np.concatenate((rgb, padding))
            # print(f'num_frame{num_frame}, front_rear_pad{self.front_rear_pad}, rgb shape{rgb.shape}')
            size[time_dim] = self.frame_len
            rgb = view_as_windows(rgb, window_shape=size, step=self.win_stride)# (view_num, 1, 1, 1, frame_len, C x H x W)
            rgb = np.squeeze(rgb, axis=(1, 2, 3)) # (view_num, frame_len, C x H x W)
            rgb = rgb.transpose([0,2,1,3,4]) # (view_num, C, frame_len x H x W)
            # if self.frame_len == 1:
            #     rgb = np.squeeze(rgb, time_dim+2)
            rgb=torch.from_numpy(rgb)
            # print(f'num_frame{num_frame}, rgb shape{rgb.shape}')

        if self.flow_root is not None:
            flow=flow.transpose(0,1)#(C x T x H x W)->(T x C x H x W)
            flow=flow.numpy()
            if self.gray_scale: flow=np.expand_dims(flow, axis=-1)
            size = list(flow.shape)
            if self.front_rear_pad != 0:
                size[time_dim] = self.front_rear_pad
                padding = np.zeros(size, dtype=np.float32)
                flow = np.concatenate((padding, flow, padding))
            elif (self.win_stride > 1): # and (self.win_stride == self.frame_len)
                self.flow_padding = self.rgb_padding + 1 # self.frame_len - size[time_dim] % self.win_stride
                size[time_dim] = self.flow_padding
                padding = np.zeros(size, dtype=np.float32)
                flow = np.concatenate((flow, padding))
            size[time_dim] = self.frame_len
            flow = view_as_windows(flow, window_shape=size, step=self.win_stride)# (view_num, 1, 1, 1, frame_len, C x H x W)
            flow = np.squeeze(flow, axis=(1, 2, 3)) # (view_num, frame_len, C x H x W)
            flow = flow.transpose([0,2,1,3,4]) # (view_num, C, frame_len x H x W)
            # if self.frame_len == 1:
            #     flow = np.squeeze(flow, time_dim+2)
            flow=torch.from_numpy(flow)
            flow = flow[:, :, :-1]
        # print(vid, rgb.shape, flow.shape)
        if self.multi: 
            iouLabel = 0
            label = torch.from_numpy(make_event_label_array(self.num_classes, num_frame, bpm,  hyoid,  lvc,  uesOpen,  uesClose,  lvcOff))
            # print(label.shape) # (batch, num_class, frame_len)
        else:
            IoU=calIoU(set(f_ind), SL, EL)    
            IoU_TH = self.frame_len / len(range(0, num_frame))
            iouLabel = 1 if IoU >= IoU_TH else 0

            label = torch.from_numpy(make_label_array(f_ind, SL, EL, self.num_classes))

        if 'PE' in self.sampling:
            f_ind = np.array(f_ind)
            if self.front_rear_pad != 0:
                padding = np.zeros(self.front_rear_pad, dtype=np.int16)
                f_ind = np.concatenate((padding, f_ind, padding))
            elif (self.win_stride > 1): # and (self.win_stride == self.frame_len) 
                # padding = self.frame_len - size[time_dim] % self.win_stride
                # size[time_dim] = padding
                # padding = np.zeros(size, dtype=np.float32)
                # rgb = np.concatenate((rgb, padding))
                self.ind_padding = self.frame_len - len(f_ind) % self.win_stride
                padding = np.zeros(self.ind_padding, dtype=np.int16)
                f_ind = np.concatenate((f_ind, padding))
            start_f_list = torch.tensor(view_as_windows(f_ind, window_shape=self.frame_len, step=self.win_stride)[:, 0], dtype=torch.int16)
            # print(rgb.shape, f_ind.shape, start_f_list.shape)

        if self.rgb_root is not None:
            if self.flow_root is not None:# self.input_type = 'both'
                '''rgb, flow 둘 다 넣었을 때 반환값'''
                if 'PE' in self.sampling:
                    return rgb, flow, label, vid, start_f_list
                else:
                    return rgb, flow, label, iouLabel, vid    
            else: # self.input_type = 'rgb'
                if 'PE' in self.sampling:
                    return rgb, label, vid, start_f_list
                else:
                    return rgb, label, iouLabel, vid
        elif self.flow_root is not None: # self.input_type = 'flow'
            if 'PE' in self.sampling:
                return flow, label, vid, start_f_list
            else:
                return flow, label, iouLabel, vid

    def __len__(self):
         return len(self.data)
