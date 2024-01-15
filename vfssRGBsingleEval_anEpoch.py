# _PE1stF 업데이트 됨
'''
evaluation for an epoch
input is dataloader
output: loss list, y_true list, y_pred list
data class file: vfssData_loadWhole_withSkippedOrInterpolated.py
'''
from tqdm import tqdm
import numpy as np # no random work
from skimage.util import view_as_windows
import torch
import matplotlib.pyplot as plt
import openpyxl

def eval(test_dataloader, model, labelType, front_rear_pad, frame_len, modelName, device, fill, bi, win_limit, criterion, inferroot=None):#, split_file, output_type
    
    if inferroot is not None:
        wb = openpyxl.Workbook()
        sheet = wb[wb.active.title]
        sheet.append(['patientID', 'start', 'end']) # interpolated 가져올때는 interpolation rate 곱해줘야 함
        saveName = inferroot.split("/")[-1]
        resultFile = f'{inferroot}/{saveName.split("_stride")[0]}.xlsx'
        
    model.eval() # # Primarily affects layers such as BatchNorm or Dropout.
    if fill is not None:
        tile_num = fill//frame_len
        partial_num = fill%frame_len
    y_true, y_score, loss  = [], [], 0.0
    for data in tqdm(test_dataloader):
        if test_dataloader.dataset.sampling == 'PE':
            inputs, labels, iouLabel, patientID, start_f_list = data
            start_f_list = start_f_list[0]
            # print(start_f_list.size())
        else:
            inputs, labels, iouLabel, patientID = data
        inputs=inputs[0] # (view_num, C, frame_len4dataset, H x W)
        patientID = patientID.item()
        labels = labels[0] # (num_classes, video_length)
        labels = labels[1].long() # (video_length)
        # idx_l = list(range(len(iou_l))) 
        # print('output from data loader: inputs-',inputs.shape, ', label-', labels.shape)
        loop = inputs.size(0) // win_limit
        if inputs.size(0) % win_limit: loop = loop+1
        # print(inputs.shape, labels.shape) #torch.Size([6, 3, 13, 224, 224]) torch.Size([66])
        prob_l = None
        for i in range(loop):
            sub_input = inputs[i*win_limit:min(inputs.size(0), i*win_limit+win_limit)]
            if test_dataloader.dataset.sampling == 'PE':
                start_f = start_f_list[i*win_limit:min(inputs.size(0), i*win_limit+win_limit)]
                start_f = start_f.to(device)
                # print(start_f.size())

            if labelType == 'frameLabel':
                sub_label = labels[i*win_limit*frame_len:min(labels.size(0), i*win_limit*frame_len+win_limit*frame_len)].to(device)
            else:
                sub_label = labels[i*win_limit:min(inputs.size(0), i*win_limit+win_limit)].to(device)
            
            # print(f'sub input shape: {sub_input.shape}')
            # print(f'sub input segment indices: {i*win_limit} to {min(inputs.size(0), i*win_limit+win_limit)}')
            # print(f'sub label shape: {sub_label.shape}')
            # print(f'label segment indices: {i*win_limit*frame_len} to {min(labels.size(0), i*win_limit*frame_len+win_limit*frame_len)}')
            if len(sub_input.size()) == 4: sub_input = sub_input.unsqueeze(0) # 배치가 1인경우 배치 생성
            # print(sub_input.shape) # (stack_num, channel, frame_len, H, W)
            if ('vgg' in modelName) and (frame_len == 1): sub_input = sub_input.squeeze(2)
            elif fill is not None:
                sub_input = sub_input.numpy()
                sub_input = np.transpose(sub_input, (2,1,0,3,4))
                sub_input = np.tile(sub_input, (tile_num, 1, 1, 1, 1))
                if partial_num != 0: sub_input=np.concatenate((sub_input, sub_input[:partial_num]))
                sub_input = np.transpose(sub_input, (2, 1, 0, 3, 4))
                sub_input = torch.from_numpy(sub_input)
            sub_input = sub_input.to(device)
            # print(sub_input.size())
            with torch.no_grad(): # test니까 gradient안흐르게 model에 넣어보기. 아직 TDN 구현 안됨.
                if bi:# 0110, 00100
                    frame_prob = model(sub_input[:, :, :frame_len], sub_input[:, :, frame_len-1:])
                elif test_dataloader.dataset.sampling == 'PE':
                    frame_prob = model(sub_input, start_f)
                else:
                    frame_prob = model(sub_input)
                # print(frame_prob.size())
                if labelType == 'frameLabel':
                    frame_prob = torch.flatten(frame_prob, start_dim=0, end_dim=1)#torch.Size([(batch) x (frame_len), num_classes=2])
                    if front_rear_pad != 0:
                        frame_prob = frame_prob[:-front_rear_pad]
                if modelName=='TDN': frame_prob = frame_prob.squeeze(1)
                # print(frame_prob)
                # print('before cutting: ', frame_prob.shape)
                frame_prob = frame_prob[:len(sub_label)]
                # print('after cutting: ', frame_prob.shape)
                if criterion is not None:
                    loss = loss + criterion(frame_prob, sub_label).item()*frame_prob.size(0) # make it 'reduction = sum'
                frame_prob = torch.nn.Softmax(dim=1)(frame_prob)# size: (stack_num, num_classes)
                frame_prob = torch.max(frame_prob, 1)[1]#.squeeze()# size: (stack_num)

            if prob_l is None: 
                prob_l = frame_prob.tolist()
            elif type(frame_prob.tolist())==int:
                prob_l.append(frame_prob.item())
            else: prob_l.extend(frame_prob.tolist())
            # print('predicted length', len(prob_l))

        y_true.extend(labels)
        y_score.extend(prob_l)
        # print(len(y_true), len(y_score))
        if inferroot is not None:
            plot_prob_l = prob_l[:len(labels)] # cut off paddding part...used when labelType frameLabel
            idx_l = list(range(len(labels))) 
            plt.plot(idx_l, labels, idx_l, plot_prob_l)
            plt.savefig(inferroot+f'/{patientID}.png')
            plt.clf()
            indices = [i for i, e in enumerate(plot_prob_l) if e == 1]
            if (indices != []) and sheet is not None: sheet.append([patientID, indices[0], indices[-1]])
    
    if inferroot is not None: wb.save(resultFile)
    if criterion is not None: loss = loss/len(y_true)
    return y_true, y_score, loss 