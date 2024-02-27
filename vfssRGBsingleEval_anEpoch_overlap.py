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
from sklearn import metrics
import timeit
def eval(test_dataloader, model, labelType, front_rear_pad, frame_len, modelName, device, fill, bi, win_limit, criterion, inferroot=None, target_layer=None):#, split_file, output_type
    
    if inferroot is not None:
        wb = openpyxl.Workbook()
        sheet = wb[wb.active.title]
        sheet.append(['patientID', 'boundary tuple list', 'GT', 'accuracy', 'f1', 'ap', 'infer_time', 'frame_len']) # interpolated 가져올때는 interpolation rate 곱해줘야 함
        saveName = inferroot.split("/")[-1]
        resultFile = f'{inferroot}/{saveName.split("_stride")[0]}.xlsx'
        
    model.eval() # # Primarily affects layers such as BatchNorm or Dropout.
    if fill is not None:
        tile_num = fill//frame_len
        partial_num = fill%frame_len
    y_true, y_score, loss  = [], [], 0.0
    if labelType == 'frameLabel': win_stride = test_dataloader.dataset.win_stride
    for data in tqdm(test_dataloader):
        start_time = timeit.default_timer()
        if 'PE' in test_dataloader.dataset.sampling:
            if test_dataloader.dataset.flow_root is not None:
                rgb, flow, labels, patientID, start_f_list = data
                rgb = rgb[0]
                flow = flow[0]
                # print('rgb: ', rgb.size(), 'flow: ', flow.size())
            else:
                inputs, labels, patientID, start_f_list = data
                inputs=inputs[0] # (view_num, C, frame_len4dataset, H x W)
            start_f_list = start_f_list[0]
            # print(start_f_list.size())
        else:
            inputs, labels, iouLabel, patientID = data
            inputs=inputs[0] # (view_num, C, frame_len4dataset, H x W)
        patientID = patientID.item()
        labels = labels[0] # (num_classes, video_length)
        labels = labels[1].long().to(device) # (video_length)
        # idx_l = list(range(len(iou_l))) 
        # print('output from data loader: inputs-',inputs.shape, ', label-', labels.shape)
        if test_dataloader.dataset.flow_root is not None:
            input_size = rgb.size(0)
        else:
            input_size = inputs.size(0)
        loop = input_size // win_limit
        if input_size % win_limit: loop = loop+1
        # print(inputs.shape, labels.shape) #torch.Size([6, 3, 13, 224, 224]) torch.Size([66])
        prob_l = None
        for i in range(loop):
            if test_dataloader.dataset.flow_root is not None:
                # print('rgb:', rgb.size(), 'flow:', flow.size(), 'labels:', labels.size())
                sub_rgb = rgb[i*win_limit:min(input_size, i*win_limit+win_limit)]
                sub_flow = flow[i*win_limit:min(flow.size(0), i*win_limit+win_limit)]
            else:
                sub_input = inputs[i*win_limit:min(input_size, i*win_limit+win_limit)]
            if 'PE' in test_dataloader.dataset.sampling:
                start_f = start_f_list[i*win_limit : min(input_size, i*win_limit+win_limit)]
                start_f = start_f.to(device)
                # print(start_f.size())
            # if labelType == 'frameLabel':
            #     # win_stride = frame_len - 7 # resNet 첫 conv layer kernel이 7x7x7이라서 test 시 7 frames 씩 overlap 되게하기
            #     # 그럴게 아니고, I3D 실험에서 input 단위가 13이었고, 정 가운데 프레임 맞추는 거니까, 앞의 6개 프레임을 보고 맞춤. 6 frame 씩 overlap 되게 하기!
            #     sub_label = labels[i*win_limit*win_stride : min(labels.size(0), i*win_limit*win_stride + win_limit*frame_len)].to(device)
            # else:
            #     sub_label = labels[i*win_limit : min(input_size, i*win_limit+win_limit)].to(device)
            
            # print(f'sub input shape: {sub_input.shape}')
            # print(f'sub input segment indices: {i*win_limit} to {min(inputs.size(0), i*win_limit+win_limit)}')
            # print(f'sub label shape: {sub_label.shape}')
            # print(f'label segment indices: {i*win_limit*frame_len} to {min(labels.size(0), i*win_limit*frame_len+win_limit*frame_len)}')
            
            if test_dataloader.dataset.flow_root is not None:
                if sub_rgb.ndim == 4:
                    sub_rgb = sub_rgb.unsqueeze(0)
                    sub_flow = sub_flow.unsqueeze(0)
            elif sub_input.ndim == 4: 
                sub_input = sub_input.unsqueeze(0) # 배치가 1인경우 배치 생성
            # print(sub_input.shape) # (stack_num, channel, frame_len, H, W)
            if ('vgg' in modelName) and (frame_len == 1): sub_input = sub_input.squeeze(2)
            elif fill is not None:
                sub_input = sub_input.numpy()
                sub_input = np.transpose(sub_input, (2,1,0,3,4))
                sub_input = np.tile(sub_input, (tile_num, 1, 1, 1, 1))
                if partial_num != 0: sub_input=np.concatenate((sub_input, sub_input[:partial_num]))
                sub_input = np.transpose(sub_input, (2, 1, 0, 3, 4))
                sub_input = torch.from_numpy(sub_input)
            if test_dataloader.dataset.flow_root is not None:
                sub_rgb = sub_rgb.to(device)
                sub_flow = sub_flow.to(device)
            else:
                sub_input = sub_input.to(device)
            # print(sub_input.size())
            with torch.no_grad(): # test니까 gradient안흐르게 model에 넣어보기. 아직 TDN 구현 안됨.
                if bi:# 0110, 00100
                    frame_prob = model(sub_input[:, :, :frame_len], sub_input[:, :, frame_len-1:])
                elif 'PE' in test_dataloader.dataset.sampling:
                    if test_dataloader.dataset.flow_root is not None:
                        frame_prob = model(sub_rgb, sub_flow, start_f)
                    else:
                        frame_prob = model(sub_input, start_f)
                else:
                    frame_prob = model(sub_input)
                # print(frame_prob.size())
                if modelName=='TDN': frame_prob = frame_prob.squeeze(1)

            if labelType == 'frameLabel':
                if win_stride < frame_len:
                    if prob_l is None: 
                        prob_l = frame_prob[0]
                        if frame_prob.size(0) > 1:
                            frame_prob = frame_prob[1:]
                            for s in range(len(frame_prob)):
                                # 겹치는 부분은 다음 segment 예측하는 데 쓰고 버리기
                                # prob_l[-(frame_len-win_stride):] = torch.sum(torch.stack((prob_l[-(frame_len-win_stride):], frame_prob[s][:(frame_len-win_stride)])), dim=0)/(frame_len-win_stride)
                                prob_l = torch.cat((prob_l, frame_prob[s][(frame_len-win_stride):]))
                    else:
                        for s in range(len(frame_prob)):
                            # 겹치는 부분은 다음 segment 예측하는 데 쓰고 버리기
                            # prob_l[-(frame_len-win_stride):] = torch.sum(torch.stack((prob_l[-(frame_len-win_stride):], frame_prob[s][:(frame_len-win_stride)])), dim=0)/(frame_len-win_stride)
                            prob_l = torch.cat((prob_l, frame_prob[s][(frame_len-win_stride):]))
                    
                    if i == (loop-1):
                        # pad = test_dataloader.dataset.rgb_padding
                        prob_l = prob_l[:labels.size(0)]
                else:
                    frame_prob = torch.flatten(frame_prob, start_dim=0, end_dim=1)#torch.Size([(batch) x (frame_len), num_classes=2])
                    if i == (loop-1):
                        # pad = test_dataloader.dataset.rgb_padding
                        non_pad = labels.size(0) - i*win_limit*win_stride
                        if non_pad > 0: 
                            frame_prob = frame_prob[:non_pad]#[:len(sub_label)] # 잘못 잘리는 건 아닌 지 확인
                    
                    if prob_l is None: 
                        prob_l = frame_prob#.tolist()
                    elif type(frame_prob.tolist())==int: # if win_stride < frame_len, it'll never be this case
                        prob_l.append(frame_prob.item())
                    else: 
                        prob_l = torch.cat((prob_l, frame_prob)) # prob_l.extend(frame_prob.tolist()) 
            else:
                if prob_l is None: 
                    prob_l = frame_prob#.tolist()
                elif type(frame_prob.tolist())==int: # if win_stride < frame_len, it'll never be this case
                    prob_l.append(frame_prob.item())
                else: 
                    prob_l = torch.cat((prob_l, frame_prob)) # prob_l.extend(frame_prob.tolist()) 
                # print('predicted length', len(prob_l))
        if criterion is not None:
            loss = loss + criterion(prob_l, labels).item()*frame_prob.size(0) # make it 'reduction = sum'

        prob_l = torch.nn.Softmax(dim=1)(prob_l)# size: (stack_num, num_classes)
        prob_l = torch.max(prob_l, 1)[1]
        if inferroot is not None:
            arg_labels = torch.argwhere(labels)
            args_prob_l = torch.argwhere(prob_l)
            prob_bound = []
            # prior = arg_labels[0].item() - 2
            start=None
            for i in range(len(args_prob_l)):
                if start is None:
                    start=args_prob_l[i]
                if i == len(args_prob_l)-1 or args_prob_l[i]+1 != args_prob_l[i+1]:
                    if start != args_prob_l[i]:
                        prob_bound.append((start, args_prob_l[i]))
                    else:
                        prob_bound.append((start,))
                    start=None
        prob_l = prob_l.tolist()#.squeeze()# size: (stack_num)
        labels = labels.tolist()
        y_true.extend(labels)
        y_score.extend(prob_l)
        # print(len(y_true), len(y_score))
        if inferroot is not None:
            # plot_prob_l = prob_l[:len(labels)] # cut off paddding part...used when labelType frameLabel
            idx_l = list(range(len(labels))) 
            plt.plot(idx_l, labels, '#1f77b4', label=f'GT {min(arg_labels).item()} : {max(arg_labels).item()}')
            plt.plot(idx_l, prob_l, '#ff7f0e', label='Pred')
            # if len(args_prob_l) > 0:
            #     plt.plot(idx_l, prob_l, '#ff7f0e', label='Pred')
            #     # plt.xticks() # pad=6, labelcolor='#1f77b4' 안됨
            #     # plt.xticks([min(arg_labels).item(), max(arg_labels).item()], [min(args_prob_l).item(), max(args_prob_l).item()]) # pad=15, labelcolor='#ff7f0e' 안됨
            #     # plt.xticks(label_bound+prob_bound, linespacing=3)
            #     # plt.xticks([min(args_prob_l).item(), max(args_prob_l).item()]) # pad=15, labelcolor='#ff7f0e' 안됨
            # else:
            #     plt.plot(idx_l, prob_l, '#ff7f0e', label='Pred')
            plt.legend()
            plt.savefig(inferroot+f'/{patientID}.png')
            plt.clf()
            # indices = [i for i, e in enumerate(prob_l) if e == 1] # plot_prob_l
            # if (indices != []) and sheet is not None: sheet.append([patientID, indices[0], indices[-1]])
            accuracy = round(metrics.accuracy_score(labels, prob_l).item(), 4)
            F1_score = round(metrics.f1_score(labels, prob_l, pos_label=1, average='binary').item(), 4)
            ap = round(metrics.average_precision_score(labels, prob_l), 4)
            stop_time = timeit.default_timer()
            # ['patientID', 'boundary tuple list', 'GT', 'accuracy', 'f1', 'ap', 'infer_time', 'frame_len']
            sheet.append([patientID, str(prob_bound), str(arg_labels), accuracy, F1_score, ap, str(stop_time - start_time), len(labels)])

    if inferroot is not None: wb.save(resultFile)
    if criterion is not None: loss = loss/len(y_true)
    return y_true, y_score, loss 