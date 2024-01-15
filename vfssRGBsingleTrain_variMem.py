'''
conda create --name syj_scnn python=3.8.5
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

conda install tqdm


conda install pandas

conda install openpyxl

conda install psutil

conda install matplotlib

pip install sklearn

conda install -c conda-forge tensorboard


conda install -c conda-forge torchinfo
'''
# train_vgg_res3d_only.py, VariTrain_earlyStop.py 섞음. VariTrain_earlyStop.py는 frame_length 가변적으로 할 코드가 있어서 지우면 안됨.
def train_model(split_file, rgb_root, flow_root, num_epochs, frame_len, sampling, train_batch, val_batch, 
num_classes, lr, last_model, modelName, n_epochs_stop, loss_type, fill, bi, win_limit, pad, gray_scale=False):
    import os
    import sys
    import timeit
    from tqdm import tqdm
    #import pickle
    import psutil
    #from sklearn.model_selection import KFold
    from sklearn import metrics
    #import socket
    import conv3dVari_each as conv3dVari
    
    from earlyStopping import EarlyStopping
    #from sklearn.model_selection import KFold
    from skimage.util import view_as_windows
    from returnModel_fixed import returnModel
    # Nested tensor로 dataloader 만드는 법 알아보기. 길이가 다 달라서 batch 로 안묶임
    # loss 들어갈 때 label갯수랑 model output 갯수랑 다른거 같음 오류남  
    if ('Skipped' in split_file) or ('interpolated' in split_file) :
        # print('it is vfssData_withSkippedOrInterpolated')
        from vfssData_loadWhole_withSkippedOrInterpolated import VFSS_data
    else:
        from vfssData_loadWhole import VFSS_data

    # https://hoya012.github.io/blog/reproducible_pytorch/
    # SEED = 777
    import random
    # random.seed(SEED)
    import numpy as np
    # np.random.seed(SEED)

    import torch
    import torch.utils.data as data_utl
    from torch.utils.tensorboard import SummaryWriter
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # from torchinfo import summary

    save_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 파일이 있는 디렉토리
    parentdir = os.path.dirname(save_dir)

    splitFileName = split_file.split('/')[-1]
    if not os.path.isfile(split_file):
        split_file = os.path.join(save_dir, splitFileName)
        if not os.path.isfile(split_file):
            split_file = os.path.join(parentdir, splitFileName)
            if not os.path.isfile(split_file): sys.exit('specify the whole path of the split file')
    splitFileName = splitFileName.split('.')[0]

    if flow_root is not None:
        if not os.path.exists(flow_root):
            flowName = flow_root.split('/')[-1]
            flow_root = os.path.join(save_dir, flowName)
            if not os.path.exists(flow_root): 
                flow_root = os.path.join(parentdir, flowName)
                if not os.path.exists(flow_root): 
                    sys.exit('specify the flow_root directory')
    if rgb_root is not None:
        if not os.path.exists(rgb_root):
            rgbName = rgb_root.split('/')[-1]
            rgb_root = os.path.join(save_dir, rgbName)
            if not os.path.exists(rgb_root):
                rgb_root = os.path.join(parentdir, rgbName)
                if not os.path.exists(rgb_root):
                    sys.exit('specify the rgb_root directory')
        if flow_root is not None: 
            input_type='both'
            sys.exit('supported in VariTrain_both.py')
        else: # root.split('/')[-1]=='frames':
            input_type='rgb'
            in_channels=3
    elif flow_root is not None: # root.split('/')[-1]=='flows':
        input_type='flow'
        in_channels=2
        sys.exit('supported in VariTrain_earlyStop.py')
    else:
        sys.exit('need data root directory')

    print('the frame root is: ', rgb_root)    
    
    # Use GPU if available else revert to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    resume_epoch=0
    if last_model is not None:
        checkpoint = torch.load(last_model, map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        resume_epoch=checkpoint['epoch']-1
        
    # midLabelModel = ['CNN3D', 'i3d', 'VTN', 'vgg', 'efficient', 'vgg3d','resNet3D', 'resNet3Dbi', 'TDN']
    
    resize, labelType, model, optimizer, scheduler, transform, front_rear_pad, win_stride = returnModel(modelName, resume_epoch, save_dir, num_classes, lr, input_type, frame_len, bi)
    model=model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # DataParallel로 만든 state dict는 DataParallel로 만든 모델에만 넣어야 한다.
    if last_model is not None:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
    
    #모델과 tensorboard graph를 저장할 디렉토리 
    if labelType == 'label':
        folderName = 'VariModels_labelPro'
    elif labelType == 'frameLabel':
        folderName = 'VariModels_frameLabel'
    elif labelType == 'IoU':
        folderName = 'VariModels'
    os.makedirs(os.path.join(save_dir, folderName), exist_ok=True)
    #모델과 tensorboard graph를 저장할 이름
    saveName = f'{input_type.upper()}model-{modelName}_frame-{frame_len}_sample-{sampling}_split-{splitFileName}_LR-{lr}_batch-{train_batch}_loss-{loss_type}'#_iou-{iouTH}

    if bi: saveName = saveName+'_bi'
    log_dir = os.path.join(save_dir, folderName, saveName)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # print(summary(model, input_size=(train_batch, in_channels, frame_len, resize[0], resize[1]), verbose=0))
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print('Initial learning rate: ', optimizer.param_groups[0]['lr'])


    train_dataset = VFSS_data(split_file = split_file, split = 'train', rgb_root=rgb_root, flow_root=flow_root, frame_len=frame_len, win_stride=win_stride, #, bi=bi
    fill=fill, sampling=sampling, resize=resize, num_classes=num_classes, transforms=transform, gray_scale=gray_scale, front_rear_pad=front_rear_pad)

    val_dataset = VFSS_data(split_file = split_file, split = 'val', rgb_root=rgb_root, flow_root=flow_root, frame_len=frame_len, win_stride=win_stride, # , bi=bi
    fill=fill, sampling=sampling, resize=resize, num_classes=num_classes, transforms=transform, gray_scale=gray_scale, front_rear_pad=front_rear_pad)

    num_workers=min(round(len(train_dataset)/train_batch+0.4), int(psutil.cpu_count()/8))
    print('train num worker: ', num_workers)
    train_dataloader = data_utl.DataLoader(train_dataset, batch_size=train_batch, shuffle=True, num_workers=num_workers)

    num_workers=min(round(len(val_dataset)/val_batch+0.4), int(psutil.cpu_count()/8))
    print('val num worker: ', num_workers)
    val_dataloader = data_utl.DataLoader(val_dataset, batch_size=val_batch, num_workers=num_workers)
    
    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}

    if n_epochs_stop is not None: early_stopping = EarlyStopping(patience = n_epochs_stop, verbose = False, path=os.path.join(save_dir, folderName, saveName)+'.pth.tar')
    
    if loss_type=='CE':
        criterion = torch.nn.CrossEntropyLoss().to(device)  # standard crossentropy loss for classification
    else:#labelType == 'frameLabel' or frame_len==1:
        weight = torch.tensor([1,3])# the loss weight when num_classes==2
        weight = weight/weight.sum()
        criterion = conv3dVari.PenaltyLoss(torch.nn.CrossEntropyLoss(weight=weight)).to(device)
    
    # HNS=[] #hard negative sample data list
    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0
            y_true=[]
            y_score=[]

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':  
                model.train()
            else:
                model.eval()
                TP = 0.0 
                FN = 0.0
                TPFP = 0.0
                TNFN = 0.0

            for data in tqdm(trainval_loaders[phase]):
                inputs, labels, iouLabel, patientID = data
                # if load_whole:
                #     f_ind, SL, EL, IoUs, patientID = data
                #     # print(f_ind)
                #     # print(f_ind.tolist())
                #     # print(f_ind[0])
                #     # print(f_ind[0].tolist())
                #     inputs = conv3dVari.load_rgb_frames(root, patientID.item(), f_ind[0].tolist(), resize)
                #     # print(inputs.shape)#(66, 512, 512, 3)
                #     inputs = conv3dVari.video_to_tensor(inputs)
                #     # print(inputs.size())#torch.Size([3, 66, 512, 512])
                #     inputs = torch.unsqueeze(inputs, 0)
                #     # print(inputs.size())#torch.Size([1, 3, 66, 512, 512])
                #     labels = conv3dVari.make_label_array(f_ind[0].tolist(), SL, EL, num_classes)
                #     # print(labels.shape)#(2, 66)
                #     labels = torch.from_numpy(labels)
                #     # print(labels.size())#torch.Size([2, 66])
                #     labels = torch.unsqueeze(labels, 0)
                # else:
                #     inputs, labels, iouLabel, patientID = data
                inputs=inputs[0]# get rid of batch dimension
                labels = labels[0]# get rid of batch dimension
                # print(inputs.size(), labels.size())# [windows=framesInVid, channel, frames_len, H, W], [num_classes=2, framesInVid=windows]
                if inputs.size(0) > win_limit:
                    loop = inputs.size(0) // win_limit
                    if inputs.size(0) % win_limit:
                        loop = loop+1
                else:
                    loop = 1
                for i in range(loop):
                    sub_input = inputs[i*win_limit:min(inputs.size(0), i*win_limit+win_limit)]
                    sub_labels = labels[:, i*win_limit:min(inputs.size(0), i*win_limit+win_limit)]
                    # move inputs and labels to the device the training is taking place on
                    sub_input = sub_input.to(device)#Size([batch, RGB, frame_len, H, W])
                    if (modelName == 'vgg') or 'efficient' in modelName:
                        sub_input = sub_input.permute(0, 2, 1, 3, 4).contiguous()
                        sub_input = torch.flatten(sub_input, start_dim=0, end_dim=1)
                        #Size([(batch=1)*frame_len, RGB, H, W])
                    elif modelName=='TDN':# make Size([batch, frame_len, RGB, W, H])
                        sub_input = sub_input.permute(0, 2, 1, 4, 3).contiguous()

                    optimizer.zero_grad()
                    # print('totoal', inputs.size(), ', forward', inputs[:, :, :frame_len].size(), ', backward', inputs[:, :, frame_len-1:].size())
                    if phase == 'train':
                        if bi:# frame_len == 5 인 경우, 00001+10000 => 000010000 이므로.
                            outputs = model(sub_input[:, :, :frame_len], sub_input[:, :, frame_len-1:])
                        else:
                            outputs = model(sub_input)
                    else:
                        with torch.no_grad():
                            if bi:
                                outputs = model(sub_input[:, :, :frame_len], sub_input[:, :, frame_len-1:])
                            else:
                                outputs = model(sub_input)

                    if torch.isnan(torch.sum(outputs)):
                        sys.exit("There is NaN in model's output")
                        
                    if labelType == 'frameLabel': # outputs: torch.Size([batch, frame_len, num_classes)
                        outputs=torch.flatten(outputs, start_dim=0, end_dim=1)
                        if phase == 'val' and front_rear_pad != 0:# if val_batch == 1: val_load_whole = True
                            outputs = outputs[:-front_rear_pad] # Do I really need this?
                    else:
                        if ('i3d' in modelName) and (len(outputs.size())==3) and (outputs.size(-1)==1): # 얘는 front_rear_pad 안 빼도 되나
                            outputs = outputs.squeeze(2)
                        elif modelName=='TDN': outputs = outputs.squeeze(1)
                        # print(outputs.size())#torch.Size([batch=10, num_classes=2])                  

                    probs = torch.nn.Softmax(dim=1)(outputs) #torch.Size([batch=10, num_classes=2])
                    preds = torch.max(probs, 1)[1] #torch.Size([frame_len])
                    # print(sub_labels)
                    label = sub_labels[1].long().to(device)

                    print(outputs.shape, label.shape)

                    if loss_type=='CE':
                        loss = criterion(outputs, label)
                    else:#labelType == 'frameLabel' or frame_len==1:
                        loss = criterion(outputs, label, preds)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    with torch.no_grad():
                        # if epoch > 100 and loss.item() > 0.5:
                        #     #before = psutil.virtual_memory().percent
                        #     HNS.append((inputs, labels, iouLabel, patientID)) #이거 하나에 310 #copy.deepcopy(
                        #     #print(sys.getsizeof(HNS))
                        #     #after = psutil.virtual_memory().percent
                        #     #difference = after - before
                        #     #later = after + difference
                        #     #print(before, after,difference, later)
                        #     #print(len(HNS), psutil.virtual_memory().percent, sys.getsizeof(HNS))
                        #     print(psutil.virtual_memory().percent)
                        #     if psutil.virtual_memory().percent>90:#좀더 자동화 하는 법 알아보기
                        #         with open(os.path.join(save_dir, 'VariModels', saveName + '_epoch-' + str(epoch) + '.pkl'), 'wb') as f:
                        #             pickle.dump(HNS, f)
                        #         HNS=[]
                        running_loss += loss.item() * outputs.size(0)
                        running_corrects += torch.sum(preds == label.data) 
                        # preds = preds.clone().detach()
                        y_true.extend(label.tolist())
                        y_score.extend(preds.tolist())
                        if phase == 'val':
                            TP += torch.logical_and(preds, label).sum()
                            TPFP += preds.sum()
                            TNFN += (preds.size(0) - preds.sum())
                            FN += torch.logical_xor(preds, label).sum() + torch.logical_and(preds, label).sum() - preds.sum()


            ##############################################################################################################
            # training or validation step finished 
            ##############################################################################################################
            epoch_loss = running_loss / len(y_true)
            epoch_acc = running_corrects.double() / len(y_true)
            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, num_epochs, epoch_loss, epoch_acc))
            ap = metrics.average_precision_score(y_true, y_score)#AP는 Recall에 따른 Precision 곡선 아래 면적
            #현재 클래스 하나만 detect하는 것이므로, mAP==AP

            if phase=='val':
                recall = TP / (TP + FN)
                precision = TP / TPFP
                F1_score = 2*precision*recall/(precision+recall)
                #FNR = FN/(FN+TP)#False Negative Rate
                fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                zero_ratio = TNFN / (len(y_true))
                one_ratio = TPFP / (len(y_true))
                print("recall: {}".format(round(recall.item(), 3)))
                print("precision: {}".format(round(precision.item(), 3)))
                print("F1_score: {}".format(round(F1_score.item(), 3)))
                print("AUC: {}".format(round(auc, 3)))
                print("AP: {}".format(round(ap, 3)))
                print("zero_ratio: {}, one_ratio: {}".format(zero_ratio, one_ratio))

            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                if 'ReduceLROnPlateau' not in str(scheduler) and ('Adam' not in str(optimizer)):
                    scheduler.step()
                train_loss = epoch_loss
                train_acc = epoch_acc
                train_ap = ap

            if phase=='val':
                if 'ReduceLROnPlateau' in str(scheduler):
                    scheduler.step(epoch_loss)
                writer.add_scalars(f'{modelName}/loss', {'train': train_loss, 'val': epoch_loss}, epoch)
                writer.add_scalars(f'{modelName}/accuracy', {'train': train_acc, 'val': epoch_acc}, epoch)
                writer.add_scalars(f'{modelName}/average_precision', {'train': train_ap, 'val': ap}, epoch)
                writer.add_scalar(f'{modelName}/learning_rate', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar(f'{modelName}_val/output_one_ratio', one_ratio, epoch)
                writer.add_scalar(f'{modelName}_val/recall', recall, epoch)
                writer.add_scalar(f'{modelName}_val/precision',precision, epoch)
                writer.add_scalar(f'{modelName}_val/F1_score', F1_score, epoch)
                writer.add_scalar(f'{modelName}_val/AUC', auc, epoch)

                if n_epochs_stop is not None:
                    early_stopping(epoch_loss, model, epoch, optimizer)
                    if early_stopping.counter==0: val_acc = epoch_acc
                    if early_stopping.early_stop:
                        # print("Early stopping. Saved model is from epoch ", epoch+1-n_epochs_stop)
                        print("the saved model's validation accuracy was ", val_acc)
                        print("the saved mode's name is ", os.path.join(save_dir, folderName, saveName)+'.pth.tar')
                        break
            ##############################################################################################################
            # an epoch finished
            ##############################################################################################################
        if n_epochs_stop is not None and early_stopping.early_stop: 
            writer.close()
            break
    if n_epochs_stop is None: writer.close()
    '''
    if HNS != []:
        with open(os.path.join(save_dir, folderName, saveName)+'.pkl', 'wb') as f:
            pickle.dump(HNS, f)
    '''

if __name__ == '__main__':
    # os.path.dirname(os.path.abspath(__file__)) # 현재 파일이 있는 디렉토리
    # os.path.join(os.path.dirname(os.path.abspath(__file__)), '../frames')
    # os.path exists
    # os.isfile
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_file', type=str, help='split file name', default='bpm2uesClose.xlsx')
    #server2:/home/DATA/syj/vfss/C3D_VGG/bpm2ues_close_out.xlsx #local: C:/Users/singku/Downloads/vfss/C3D_VGG/bpm2ues_close_out.xlsx 
    # #server4: /DATA/jsy/vfss/bpm2ues_close_out.xlsx #lab: /home/DATA/syj/vfss/bpm2ues_close_out.xlsx
    parser.add_argument('--rgb_root', type=str, help='full path of the frames folder', default= 'frames')
    #server2:/home/DATA/syj/vfss/frames #local: C:/Users/singku/Downloads/vfss/frames #server4: /DATA/jsy/vfss/frames #lab: /home/DATA/syj/frames
    parser.add_argument('--flow_root', type=str, default=None, help='full path of the flows folder')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs for training', default=100000)
    parser.add_argument('--train_batch', type=int, help='batch size', default=1)
    parser.add_argument('--val_batch', type=int, help='batch size', default=1)
    parser.add_argument('--last_model', type=str, help='last model you wanna resume', default=None)
    parser.add_argument('--num_classes', type=int, help='number of action classes including the background', default = 2)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-5)
    parser.add_argument('--frame_len', type=int, default=11)#if 100, load_whole, then, batches are 1 for each, automatically
    parser.add_argument('--pad', type=int, default=None, help='fill the length with zero. 16 for resNet3D')
    parser.add_argument('--fill', type=int, default=None, help='fill the length with iteration. 16 for resNet3D')
    parser.add_argument('--bi', type=bool, default=False, help='whether train the model bidirectional. works only when model is resNet3D')
    parser.add_argument('--sampling', type=str, help='there is no type', default='loadWhole')
    parser.add_argument('--modelName', type=str, default='resNet3D')#frame_len must be 1 when the model is vgg3d or vgg
    parser.add_argument('--n_epochs_stop', type=int, help='early stopping threshold', default=20)
    parser.add_argument('--loss_type', type=str, help='custom or CE', default = 'CE')
    parser.add_argument('--win_limit', type=int, default=20, help='maximum frame batch that can be loaded on a GPU')
    args = parser.parse_args()

    train_model(bi = args.bi, train_batch=args.train_batch, val_batch=args.val_batch, num_epochs=args.num_epochs, last_model=args.last_model, fill = args.fill, 
    frame_len = args.frame_len, sampling = args.sampling, modelName = args.modelName, split_file=args.split_file, loss_type = args.loss_type, lr=args.lr, 
    rgb_root = args.rgb_root, flow_root = args.flow_root, num_classes=args.num_classes, n_epochs_stop=args.n_epochs_stop, pad = args.pad, win_limit=args.win_limit)
