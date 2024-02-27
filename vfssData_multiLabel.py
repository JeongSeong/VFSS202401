'''Event detection의 경우
split file에는 환자별 train, val, test여부,,,  event별 일어나는 프레임 번호
label 생성은 행이 Event, 열이 frame들
event도 argparse로 받기.. 숫자로 하자.

makeSplitFile-withSkippedMultiLabeled.py로 생성한 파일 (Multilabeled-8-2.xlsx) 사용
"C:/Users/singku/Downloads/vfss/C3D_VGG/makeSplitFile-withSkippedMultiLabeled.py"
'''
''' Phase localization의 경우
split file에는 환자별 train, val, test여부
label 생성은 행은 frame순서, 열은 구간명인 행렬에서 해당 프레임이 구간에 속하면 1, 아니면 0.
구간도 argparse로 받기.. 숫자로 하자.
'''
import torch
import numpy as np
import random
import pandas as pd
import copy
from vfssData_withSkippedOrInterpolatedPreLoaded import load_rgb_frames, load_flow_frames
import multiprocessing
from tqdm import tqdm
'''
SL, EL 리스트를 인덱스로 순회하며...인덱스가 구간명
행은 frame순서, 열은 구간명인 행렬에서 해당 프레임이 구간에 속하면 1, 아니면 0.
모델 output도 마찬가지로 나와야 함? 그럼 frame level label만 가능한데..?
그럼 가능한 모델들은 vgg, GRU붙은것들,,,,,밖에 안될거 같은데.? 아니야 생각을 해보자.........
label type 모델들에 프레임을 넣을 때,, TDN에 넣는거랑 비슷하게 하면 되지 않을까?
frame
frame
frame
.           ===>>   0fr fra ram ame me0 을 세로로 붙여서 batch 처럼 되면 순서대로 
.                   예측하고, 쪼갠만큼 다시 붙여서 loss에 넣으면 ?
.                   한 환자의 frame들을 서로 관련없게, 다른 환자와 같이 예측하게 됨.

그냥 환자 프레임 통째로 batch 1 씩 학습할까.
frame 최대 길이에 맞춰서 frame들을 구간 내 에서 중복되게 하여 늘릴까. 
그러면 한 구간 안에서도 몇 번째 frame을 늘릴 지 정해야 한다.
그냥 맨 첫 프레임이나 맨 마지막 프레임만 늘릴까
맨 마지막 프레임만 늘려야 나중에 PAS score계산할 때 noise?역할이 없지 않을까...그건 PAS랑 같이 학습할때 생각하고.

DAIN 으로 Temporal Super Resolution
'''
def make_event_label_array(num_classes, num_frame, bpm,  hyoid,  lvc,  uesOpen,  uesClose,  lvcOff): #f_ind, SL, EL, num_classes): # num_classes 는 구간 수(7) + 1 (background)

    label = np.zeros((num_classes, num_frame), np.int8) 
    # 이벤트 프레임만 detection하는 게 너무 data imbalance해서 
    #   label[1, lvc_start-start_frame:lvc_end-start_frame+1] = 1
    #   label[0, :] = 1 - label[1, :]

    label[0, :bpm] = 1 # 액체가 턱뼈 넘어가기까지...start2bpm phase 도 대신 가능. 얘도 역순이 있을 수 없고, 굳이 순간으로 labeling안 해도 될듯. 
    '''
    label[1, hyoid] = 1 # hyoid bone 뛰는 순간...? 앞뒤 한 프레임까지 할까? weight 크게 주기!! 아무리 그래도 data imbalance가 심함
    # 아님 차라리 hyoid max burst 전후로 나누는 게 나아보임. 
    # 사진 크기를 줄이면서 hyoid 가 잘 안보여서 detection못하는 걸수도. 그럼 hyoid만 빼서 큰 사진으로 따로 학습시킬까?
    # hyoid burst 를 시작하고나서도, 그 위치가 턱뼈쪽으로 더 가고 나서 돌아오기 때문에,, burst 이후를 positive(1)로 두는 게 맞아보임
    '''
    '''
    label[0, min(bpm, hyoid):max(bpm, hyoid)+1] = 1
    label[1, min(hyoid, uesClose):max(hyoid, uesClose)+1] = 1
    label[2, min(hyoid, lvc):max(hyoid, lvc)+1] = 1
    '''
    # 프레임을 쪼개서 주는 게 아니라, 거의 다 주는거면, hyoid bone bust 전까지를 하나의 label로 해도 될것 같다
    label[3, :hyoid] = 1

    label[1, lvc:lvcOff+1] = 1 # LVC:LVC_OFF 이거는 한 phase class (LV가 붙어있는 기간)로 보는 게 나아보임. 역순이 있을 수 없는 개념
    
    # 사진 크기를 줄이면서, 잘 안보여서 성능이 살짝 낮은 것 같다. 큰사진으로 따로 학습시키는 것도 좋을 것 같다.
    # label[4, uesOpen] = 1 # 이 순간 이전까지 positive(1)로 label
    # label[5, uesClose] = 1 # 이 순간 이후부터 positive(1)로 label
    
    label[2, uesOpen:uesClose+1] = 1
    # label[6, lvcOff] = 1
    # background class 는 그냥 레이블이 없는, 죄다 0인 것으로 보는 게 맞음. 따로 1로 할당하지 마라.
    # label[0, :] = 1 - (label[1, :] | label[2, :] | label[3, :] | label[4, :] | label[5, :] | label[6, :])
    return label

def make_multiLabel_dataset(num_classes, split_file, split, rgb_root, flow_root, resize, gray_scale, transforms): # default: rgb_root=None, flow_root=None
    '''
    고른 구간에 해당하는 SL, EL 리스트를 만들어 리턴.
    split_file = 'MultiLabeled-8-2.xlsx'
    '''
    df = pd.read_excel(split_file)
    df=df.loc[df['split']==split, :]
    df.drop(columns=['split'], inplace=True)
    df.set_index('patient_id', drop=True, inplace=True)
    # print(df.head())
    df = df.values.tolist() # vid, num_frame, bpm,  hyoid,  lvc,  uesOpen,  uesClose,  lvcOff
    dataset = []
    pool = multiprocessing.Pool()
    for vid, num_frame, bpm,  hyoid,  lvc,  uesOpen,  uesClose,  lvcOff in tqdm(df): # 
        # print(vid, num_frame, bpm,  hyoid,  lvc,  uesOpen,  uesClose,  lvcOff)
        label = make_event_label_array(num_classes, num_frame, bpm,  hyoid,  lvc,  uesOpen,  uesClose,  lvcOff)
        # print(label.shape)
        f_ind = list(range(num_frame))
        
        if rgb_root is not None:
            rgb = load_rgb_frames(rgb_root, vid, f_ind, transforms, pool, resize, gray_scale)
        else:
            rgb = None

        if flow_root is not None:# self.input_type = 'both'
            flow = load_flow_frames(flow_root, vid, f_ind, transforms, pool, resize)
        else:
            flow=None
        # print(rgb.shape, flow, vid)
        dataset.append((vid, num_frame, rgb, flow, label))
    pool.close()
    return dataset

class VFSS_data(torch.utils.data.Dataset): # root, input_type 사라지고 rgb_root, flow_root 생김, fill 추가됨

    def __init__(self, split_file, split, frame_len, sampling, transforms, labelType, gray_scale=False, fill=None
    , rgb_root=None, flow_root=None, num_classes=2, resize=None, load_whole= False):

        self.split_file = split_file
        self.transforms = transforms
        self.labelType = labelType
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
        self.data = make_multiLabel_dataset(num_classes, split_file, split, rgb_root, flow_root, resize, gray_scale, transforms) 

    def __getitem__(self, index):
        vid, num_frame, rgb, flow, label = self.data[index]#, label
        # start frame, end frame으로 계산하면 비는 frame있는 영상 처리하는데에 문제가 생기니까, 
        # num_frame을 따로 계산해서 엑셀 파일에 저장해놓는 것을 구현하기
        if self.load_whole:
            f_ind = list(range(num_frame))
        elif self.frame_len == 500:
            start_f = 0
            f_ind=list(range(start_f, min(start_f+self.frame_len, num_frame)))
        elif self.sampling == 'PE':
            start_f = random.randint(0, num_frame-1)
            f_ind=list(range(start_f, min(start_f+self.frame_len, num_frame)))
        else: # SL, EL이 없다. 
            # # 최소 하나의 label이 포함된 구간 고르는 거 (label[0, :] == 0 인 부분 포함되게) 해보기!!!!
            # if self.sampling =='revised' and self.split == 'train':
            #     coin = random.randint(0,2)
            #     if coin == 2:
            #         indices = np.where(label[0]==0)
            #         first = indices[0]
            #         last = indices[-1]
            #         if self.labelType == 'frameLabel':
            #             start_f = random.randint(max(first-self.frame_len, 0), max(last-self.frame_len+1, 0))
            #         else:
            #             start_f = random.randint(first-(self.frame_len//2), last-(self.frame_len//2))
            #     else:
            #         if self.labelType == 'frameLabel':
            #             start_f = random.randint(0, max(num_frame-self.frame_len, 0))
            #         else:
            #             start_f = random.randint(-(self.frame_len//2), num_frame-(self.frame_len//2))
            # else: # frameLabel모델에 적용하는 것이 적합하다. 랜덤추출 시에는 data imbalance가 심할 수 있다.
            #     start_f = random.randint(0,max(num_frame-self.frame_len, 0))
            # f_ind=list(range(start_f, min(start_f+self.frame_len, num_frame)))
            import sys
            sys.exit('only PE sampling and load_whole are accepted')

        label = label[:, f_ind]
        if self.rgb_root is not None:
            rgb = torch.index_select(input=rgb, dim=1, index=torch.tensor(f_ind))
        if self.flow_root is not None:
            flow = torch.index_select(input=flow, dim=1, index=torch.tensor(f_ind))
        if self.load_whole: # 너무 크니까, 데이터로더에서 나온거 바로 GPU에 넣지 말기!!!
            return rgb, flow, torch.from_numpy(label), vid

        if self.frame_len != 500:
            if (not self.load_whole) and self.frame_len!=1:
                if start_f < 0: # this must be revised sampling, frame_len > 1
                    # np.zeros((num_classes, len(f_ind)), np.float32)
                    padding = list(label.shape)
                    padding[1] = -start_f
                    padding = np.zeros(padding, np.int8)
                    padding[0, :] = 1
                    label = np.concatenate((padding, label), axis=1)
                if start_f > (num_frame-self.frame_len):
                    padding = list(label.shape)
                    padding[1] = start_f+self.frame_len - num_frame
                    padding = np.zeros(padding, np.int8)
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
                if (not self.load_whole) and self.frame_len!=1:
                    if start_f < 0:
                        size = list(rgb.size())
                        size[1] = -start_f
                        rgb = torch.cat((rgb.new_zeros(size), rgb), dim=1)
                    if start_f > (num_frame-self.frame_len):
                        size = list(rgb.size())
                        size[1] = start_f+self.frame_len - num_frame
                        rgb = torch.cat((rgb, rgb.new_zeros(size)), dim=1)

            if self.flow_root is not None:
                if (not self.load_whole) and self.frame_len!=1:
                    if start_f < 0: # this must be revised sampling, frame_len > 1
                        size = list(flow.size())
                        size[1] = -start_f
                        flow = torch.cat((flow.new_zeros(size), flow), dim=1)
                    if start_f > (num_frame-self.frame_len):
                        size = list(flow.size())
                        size[1] = start_f+self.frame_len - num_frame
                        flow = torch.cat((flow, flow.new_zeros(size)), dim=1)

        # print(rgb.shape) # (batch, channel, frame_len, H, W)
        # print(label.shape) # (batch, num_class, frame_len)
        if self.rgb_root is not None:
            if self.flow_root is not None:
                '''rgb, flow 둘 다 넣었을 때 반환값'''
                if self.sampling == 'PE':
                    return rgb, flow, torch.from_numpy(label), vid, torch.tensor(start_f)
                else:
                    return rgb, flow, torch.from_numpy(label), vid    
            else: # self.input_type = 'rgb'
                if self.sampling == 'PE': # 이것만 쓰게 코딩함!
                    return rgb, torch.from_numpy(label), vid, torch.tensor(start_f)
                else:
                    return rgb, torch.from_numpy(label), vid
        else: #if self.flow_root is not None: # self.input_type = 'flow'
            if self.sampling == 'PE':
                return flow, torch.from_numpy(label), vid, torch.tensor(start_f)
            else:
                return flow, torch.from_numpy(label), vid

    def __len__(self):
         return len(self.data)
