'''
conda create --name syj_vfss_new python=3.8.5

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia

conda install tqdm

conda install pandas

conda install openpyxl

conda install matplotlib

pip install sklearn

conda install scikit-image

conda install -c conda-forge opencv

pip install grad-cam

pip install ttach
'''
# vfssRGBsingleTest.py보다, vfssRGBsingleTestBatchSize.py보다 CPU랑 Meeory 많이 잡아먹지만, 수행속도는 훨씬 빠르다
# data class file: vfssData_loadWhole_withSkippedOrInterpolated_PE1stF.py
# 수정중!!!
import os
import autoPath
# from returnModel_fixed import returnModel
from sklearn import metrics
import matplotlib.pyplot as plt
import openpyxl
from skimage.util import view_as_windows
import torch
from tqdm import tqdm
import sys
import numpy as np
from vfssRGBsingleEval_anEpoch_PE1stF import eval
def test_model(last_model, rgb_root, flow_root, split_file, num_classes, win_limit):
    if ('Skipped' in split_file) or ('interpolated' in split_file) :
        # from vfssData_loadWhole_withSkippedOrInterpolated_PE1stF import VFSS_data # server1 code
        from vfssData_loadWhole_withSkippedOrInterpolated import VFSS_data # server3 code
    else:
        from vfssData_loadWhole import VFSS_data
        sys.exit('have not tested yet')

    save_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 파일이 있는 디렉토리
    # parentdir = os.path.dirname(save_dir)

    split_file, splitFileName = autoPath.findFile(split_file, save_dir)
    if splitFileName not in last_model:
        sys.exit('model and split file mismatch')
    flow_root = autoPath.findDir(flow_root, save_dir)
    rgb_root = autoPath.findDir(rgb_root, save_dir)
    if rgb_root is not None:
        if flow_root is not None:
            input_type='both'
            from returnModel_both import returnModel
            assert modelName == 'flowMix', 'both type is only for flowMix model'
            # sys.exit('supported in VariTrain_both.py')
        else:
            input_type='rgb'
            from returnModel_fixed import returnModel
            # in_channels=3
            # root = rgb_root ############# 지울 수 있도록 하기!!!!!!!!!!
    elif flow_root is not None:
        input_type='flow'
        from returnModel_fixed import returnModel
        # in_channels=2
        # root = flow_root ############# 지울 수 있도록 하기!!!!!!!!!!
    else:
        sys.exit('need data root directory')

    last_model_split = last_model.split('/')
    #last_model은 항상 모델이 있는 폴더명부터 줘야 함
    if 'label' in last_model_split[-2]:
        labelType='label'
        folderName = 'VariModels_labelPro'
        sys.exit('only for frameLavel labelType')
    elif 'frameL' in last_model_split[-2]:
        labelType='frameLabel'
        folderName='VariModels_frameLabel'
    elif 'new50' in last_model_split[-2]:
        labelType='frameLabel'
        folderName='new50'
    else:
        sys.exit('not using now')
        # labelType='IoU'
        # folderName = 'VariModels'

    saveName = last_model_split[-1][:-8]# .pth.tar 제거
    saveNameList = saveName.split('_')
    bi=False
    fill=None
    for e in saveNameList:
        if 'model' in e:
            modelName = e
        elif 'frame' in e:
            frame_len = int(e.split('-')[-1])
        elif 'sample' in e:
            sampling = e.split('-')[-1]
        elif 'LR' in e:
            LR = e[3:]
        elif 'batch' in e:
            batch = int(e.split('-')[-1])
            if batch>win_limit: win_limit=batch
        elif e == 'bi':
            bi = True
        elif 'fill' in e:
            fill = int(e[4:])
    if fill is not None:
        tile_num = fill//frame_len
        partial_num = fill%frame_len
        # elif 'valAcc' in e:
        #     val_acc = e.split('-')[-1]

    # Use GPU if available else revert to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resume_epoch=0
    resize, labelType, model, _, __, transform, front_rear_pad, win_stride, target_layer = returnModel(modelName, resume_epoch, save_dir, num_classes, float(LR), input_type, frame_len, bi, trainVal=False)
    if '512' in sampling:
        resize = (512, 512)
        sampling = sampling.replace('512', '') 
    model=model.to(device)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    saveName = saveName+'_stride'+str(win_stride)
    inferroot = os.path.join(save_dir, folderName, saveName)
    os.makedirs(inferroot, exist_ok=True)
    os.makedirs(inferroot+'/grad_cam', exist_ok=True)# VariTest_gradCAM.py 섞기

    checkpoint = torch.load(last_model, map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
    print("Initializing weights from: {}...".format(last_model)) 
    if 'state_dict' in checkpoint.keys():
        if 'module.' in list(checkpoint['state_dict'].keys())[0]:
            from collections import OrderedDict
            checkpoint['state_dict'] = OrderedDict((k[7:], v) for k, v in checkpoint['state_dict'].items())
        model.load_state_dict(checkpoint['state_dict'])
    else: 
        model.load_state_dict(checkpoint)

    if bi:
        frame_len4dataset = frame_len*2-1
    else:
        frame_len4dataset = frame_len
    test_dataset = VFSS_data(split_file = split_file, split = 'test', rgb_root=rgb_root, flow_root=flow_root, frame_len=frame_len4dataset, win_stride=win_stride,
        fill=fill, sampling=sampling, resize=resize, num_classes=num_classes, transforms=transform, front_rear_pad=front_rear_pad) # , gray_scale=gray_scale
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=False)    
    # print('dataloader length:',len(test_dataset))
    

    y_true, y_score, _ = eval(test_dataloader, model, labelType, front_rear_pad, frame_len, modelName, device, fill, bi, win_limit, criterion=None, inferroot=inferroot, target_layer=target_layer)

    
    accuracy = metrics.accuracy_score(y_true, y_score) # corrects.double() / data_len
    recall = metrics.recall_score(y_true, y_score, pos_label=1, average='binary')# TP / (TP + FN)
    precision = metrics.precision_score(y_true, y_score, pos_label=1, average='binary')# TP / TPFP
    F1_score = metrics.f1_score(y_true, y_score, pos_label=1, average='binary')# 2*precision*recall/(precision+recall)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    accuracy = round(accuracy.item(), 3)
    recall = round(recall.item(), 3)
    precision = round(precision.item(), 3)
    F1_score = round(F1_score.item(), 3)
    auc = round(metrics.auc(fpr, tpr), 3)
    ap = round(metrics.average_precision_score(y_true, y_score), 3)
    print("Acc: {}".format(accuracy))
    print("recall: {}".format(recall))
    print("precision: {}".format(precision))
    print("F1_score: {}".format(F1_score))
    print("AUC: {}".format(auc))
    print("AP: {}".format(ap))
    plt.plot(fpr, tpr)
    plt.savefig(inferroot+f'/ROC_curve_testAcc-{accuracy}_recall-{recall}_precision-{precision}_F1score-{F1_score}_AUC-{auc}_AP-{ap}.png')
    plt.clf()
    print('the model was: ', last_model.split('/')[-1])

    resultFile = 'output.xlsx'
    if not os.path.isfile(os.path.join(save_dir, resultFile)):
        wb = openpyxl.Workbook()
        sheet = wb[wb.active.title]
        sheet.append(['LR', 'Test_acc', 'recall', 'precision', 'F1_score', 'AUC', 'AP', 'model'])
        wb.save(os.path.join(save_dir, resultFile))

    wb = openpyxl.load_workbook(resultFile)
    sheet = wb[wb.active.title] # Sheet1, Sheet
    sheet.append([LR, accuracy, recall, precision, F1_score, auc, ap, last_model.split('/')[-1][:-7]])
    wb.save(resultFile)

if __name__ == '__main__':
    import argparse
    # need to add argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_root', type=str, help='full path of the frames folder', default= None) #'frames'
    #server2:/home/DATA/syj/vfss/frames #local: C:/Users/singku/Downloads/vfss/frames #server4: /DATA/jsy/vfss/frames #lab: /home/DATA/syj/frames
    parser.add_argument('--flow_root', type=str, default=None, help='full path of the flows folder')
    parser.add_argument('--last_model', type=str, help='pth.tar file name', required=True)
    parser.add_argument('--split_file', type=str, help='split file name', default='bpm2ues_close_out.xlsx')
    parser.add_argument('--num_classes', type=int, help='number of action classes including the background', default = 2)
    # parser.add_argument('--win_stride', type=int, help='window sliding stride. at most frame_len', default=1)
    parser.add_argument('--win_limit', type=int, default=32, help='maximum frame batch that can be loaded on a GPU if it is smaller than the train batch size, win_limit is set to the train batch size')
    args = parser.parse_args()

    test_model(last_model=args.last_model, win_limit = args.win_limit, split_file=args.split_file, rgb_root = args.rgb_root, flow_root = args.flow_root, num_classes=args.num_classes) #win_stride=args.win_stride, 