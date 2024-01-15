'''
evaluation for an epoch
input is dataloader
output: loss list, y_true list, y_pred list
data class file: vfssData_loadWhole_withSkippedOrInterpolated_PE1stF.py
'''
from tqdm import tqdm
import numpy as np # no random work
from skimage.util import view_as_windows
import torch
import matplotlib.pyplot as plt
import openpyxl
# import os
# import sys
# from PIL import Image
from sklearn import metrics
import timeit
def eval(test_dataloader, model, labelType, front_rear_pad, frame_len, modelName, device, fill, bi, win_limit, criterion, inferroot=None, target_layer=None):#, split_file, output_type
    # if target_layer is not None: 
    #     save_dir = '/'.join(inferroot.split('/')[:-2])
    #     parentdir = os.path.dirname(save_dir)
    #     try:
    #         from pytorch_grad_cam import GradCAM
    #         from pytorch_grad_cam.utils.image import show_cam_on_image
    #     except:
    #         if not os.path.exists(os.path.join(save_dir, 'pytorch_grad_cam')):
    #             if not os.path.exists(os.path.join(parentdir, 'pytorch_grad_cam')):
    #                 sys.exit('cannot find pytorch_grad_cam folder')
    #             elif parentdir not in sys.path:
    #                 sys.path.insert(0, parentdir)
    #         try:
    #             from pytorch_grad_cam import GradCAM
    #             from pytorch_grad_cam.utils.image import show_cam_on_image
    #         except Exception as e:
    #             print('cannot import pytorch_grad_cam')
    #     cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device=="cuda")) # VariTest_gradCAM.py 섞기
    #     target_category = None # VariTest_gradCAM.py 섞기

    if inferroot is not None:
        wb = openpyxl.Workbook()
        sheet = wb[wb.active.title]
        sheet.append(['patientID', 'start', 'end', 'start_G', 'end_G', 'accuracy', 'f1', 'ap', 'infer_time', 'frame_len']) # interpolated 가져올때는 interpolation rate 곱해줘야 함
        saveName = inferroot.split("/")[-1]
        resultFile = f'{inferroot}/{saveName.split("_stride")[0]}.xlsx'
        
    model.eval() # # Primarily affects layers such as BatchNorm or Dropout.
    if fill is not None:
        tile_num = fill//frame_len
        partial_num = fill%frame_len
    y_true, y_score, loss  = [], [], 0.0
    for data in tqdm(test_dataloader):
        start_time = timeit.default_timer()
        if 'PE' in test_dataloader.dataset.sampling:
            if test_dataloader.dataset.flow_root is not None:
                rgb, flow, labels, patientID, start_f_list = data
                rgb = rgb[0]
                flow = flow[0]
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
        if 'poly' in str(criterion).lower():
            labels = labels.transpose(0, 1).contiguous().long()
        else: 
            labels = labels[1].long() # (video_length)
        # idx_l = list(range(len(iou_l))) 
        # print('output from data loader: inputs-',inputs.shape, ', label-', labels.shape)
        loop = inputs.size(0) // win_limit
        if inputs.size(0) % win_limit: loop = loop+1
        # print(inputs.shape, labels.shape) #torch.Size([6, 3, 13, 224, 224]) torch.Size([66])
        prob_l = None
        for i in range(loop):
            if test_dataloader.dataset.flow_root is not None:
                sub_rgb = rgb[i*win_limit:min(inputs.size(0), i*win_limit+win_limit)]
                sub_flow = flow[i*win_limit:min(inputs.size(0), i*win_limit+win_limit)]
            else:
                sub_input = inputs[i*win_limit:min(inputs.size(0), i*win_limit+win_limit)]
            if 'PE' in test_dataloader.dataset.sampling:
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
            if len(sub_input.size()) == 4: 
                if test_dataloader.dataset.flow_root is not None:
                    sub_rgb = sub_rgb.unsqueeze(0)
                    sub_flow = sub_flow.unsqueeze(0)
                else:
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
            ##VariTest_gradCAM.py 섞기#####################################################################################################
            # if target_layer is not None: 
            #     # print(sub_input.size())#torch.Size([batch, RGB, frame_len, H, W])
            #     batch_size = sub_input.size(0)
            #     cam.batch_size = batch_size
            #     grayscale_cam = cam(input_tensor=sub_input, targets=target_category, aug_smooth=True, eigen_smooth=True)
            #     # print(grayscale_cam.shape)#torch.Size([batch, H, W, frame_len])
            #     grayscale_cam = torch.from_numpy(grayscale_cam)
            #     for b in range(batch_size):
            #         os.makedirs(f'{inferroot}/grad_cam/{patientID}/{i*batch_size+b+1}', exist_ok=True)
            #         # print(grayscale_cam[b].shape) # [H, W, frame_len]
            #         grayscale_cam_b = grayscale_cam[b].permute(2, 0, 1).contiguous()
            #         # print(grayscale_cam_b.shape) # [frame_len, H, W]
            #         sub_input_b = sub_input[b].permute(1, 2, 3, 0).contiguous()
            #         # print(sub_input_b.shape) #(frame_len, H, W, RGB)
            #         for f in range(sub_input_b.size(0)):
            #             grayscale_cam_bf = grayscale_cam_b[f]
            #             # print(grayscale_cam_bf.shape) # [H, W]
            #             gray_image = np.asarray(sub_input_b[f].cpu(), dtype=np.uint8) 
            #             # print(gray_image.shape) #(H, W, RGB)
            #             visualization = show_cam_on_image(gray_image, grayscale_cam_bf, use_rgb=True)
            #             visualization = Image.fromarray(visualization)
            #             visualization.save(f'{inferroot}/grad_cam/{patientID}/{patientID}_{i*batch_size+b+1}_{f}.png')
            #             visualization.save(f'{inferroot}/grad_cam/{patientID}/{i*batch_size+b+1}/{patientID}_{i*batch_size+b+1}_{f}.png')
            ##################################################################################################################################
            if prob_l is None: 
                prob_l = frame_prob.tolist()
            elif type(frame_prob.tolist())==int:
                prob_l.append(frame_prob.item())
            else: prob_l.extend(frame_prob.tolist())
            # print('predicted length', len(prob_l))
        if 'poly' in str(criterion).lower():
            labels = labels[:, 1]
        y_true.extend(labels)
        y_score.extend(prob_l)
        # print(len(y_true), len(y_score))
        if inferroot is not None:
            plot_prob_l = prob_l[:len(labels)] # cut off paddding part...used when labelType frameLabel
            idx_l = list(range(len(labels))) 
            arg_labels = torch.argwhere(labels)
            args_prob_l = torch.argwhere(torch.tensor(plot_prob_l))
            plt.plot(idx_l, labels, '#1f77b4', label=f'GT {min(arg_labels).item()}th : {max(arg_labels).item()}th')
            plt.plot(idx_l, prob_l, '#ff7f0e', label='Pred')
            if len(args_prob_l) > 0:
                plt.xticks([min(args_prob_l).item(), max(args_prob_l).item()]) # pad=15, labelcolor='#ff7f0e' 안됨
            plt.legend()
            plt.savefig(inferroot+f'/{patientID}.png')
            plt.clf()
            indices = [i for i, e in enumerate(plot_prob_l) if e == 1]
            if sheet is not None:  #만약 positive prediction없으면 정보 추가 안됨
                accuracy = round(metrics.accuracy_score(labels, prob_l).item(), 4)
                F1_score = round(metrics.f1_score(labels, prob_l, pos_label=1, average='binary').item(), 4)
                ap = round(metrics.average_precision_score(labels, prob_l), 4)
                stop_time = timeit.default_timer()
                if (indices != []):
                    sheet.append([patientID, indices[0], indices[-1], min(arg_labels).item(), max(arg_labels).item(), accuracy, F1_score, ap, str(stop_time - start_time), len(labels)])
                else:
                    sheet.append([patientID, None, None, min(arg_labels).item(), max(arg_labels).item(), accuracy, F1_score, ap, str(stop_time - start_time), len(labels)])

    if inferroot is not None: wb.save(resultFile)
    if criterion is not None: loss = loss/len(y_true)
    return y_true, y_score, loss 