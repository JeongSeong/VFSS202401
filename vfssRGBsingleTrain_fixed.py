'''
conda create --name syj_vfss_new python=3.8.5
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia

conda install tqdm


conda install pandas

conda install openpyxl

conda install psutil

conda install matplotlib

pip install sklearn

conda install -c conda-forge tensorboard


conda install -c conda-forge torchinfo
'''
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
from returnModel_fixed import returnModel
import autoPath
#from sklearn.model_selection import KFold
from sklearn import metrics
import torch
import torch.utils.data as data_utl
from torch.utils.tensorboard import SummaryWriter
# from torchinfo import summary
import fixRandom
# train_vgg_res3d_only.py, VariTrain_earlyStop.py 섞음. VariTrain_earlyStop.py는 frame_length 가변적으로 할 코드가 있어서 지우면 안됨.
def train_model(split_file, rgb_root, flow_root, num_epochs, frame_len, sampling, train_batch, val_batch, 
num_classes, lr, last_model, modelName, n_epochs_stop, loss_type, fill, bi, pad, gray_scale=False):
    
    if ('Skipped' in split_file) or ('interpolated' in split_file) :
        # if psutil.virtual_memory().available > 4e+10:
        #     import vfssData_withSkippedOrInterpolatedPreLoaded as vfssData
        # else:
        import vfssData_withSkippedOrInterpolatedFixed as vfssData 
    else:
        # print('it is just vfssData')
        import vfssData

    save_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 파일이 있는 디렉토리

    split_file, splitFileName = autoPath.findFile(split_file, save_dir)
    flow_root = autoPath.findDir(flow_root, save_dir)
    rgb_root = autoPath.findDir(rgb_root, save_dir)
    if rgb_root is not None:
        if flow_root is not None:
            input_type='both'
            sys.exit('supported in VariTrain_both.py')
        else:
            input_type='rgb'
    elif flow_root is not None:
        input_type='flow'
    
    # Use GPU if available else revert to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resume_epoch=0
    if last_model is not None:
        checkpoint = torch.load(last_model, map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        resume_epoch=checkpoint['epoch']-1
        
    midLabelModel = ['CNN3D', 'i3d', 'VTN', 'vgg', 'efficient', 'vgg3d','resNet3D', 'resNet3Dbi', 'resNet3D_bi', 'resNet3D_fill16', 'TDN']
    
    # fixRandom.set_seed()
    # fixRandom.configure_cudnn()
    # g = torch.Generator()
    # g.manual_seed(0)
    resize, labelType, model, optimizer, scheduler, transform, _, __, ___ = returnModel(modelName, resume_epoch, save_dir, num_classes, lr, input_type, frame_len, bi)
    # if sampling == 'p512': resize=(512, 512)
    # elif sampling == 'pIoU': labelType = 'IoU'
    # ex) 
    if '512' in sampling: resize=(512, 512)
    if 'IoU' in sampling: labelType = 'IoU'
    #모델과 tensorboard graph를 저장할 디렉토리 
    if labelType == 'label':
        folderName = 'VariModels_labelPro'
    elif labelType == 'frameLabel':
        folderName = 'VariModels_frameLabel'
    elif labelType == 'IoU':
        folderName = 'VariModels'
    os.makedirs(os.path.join(save_dir, folderName), exist_ok=True)
    if fill is not None:
        modelName = modelName+'_fill'+str(fill)
    elif pad is not None:
        modelName = modelName+'_pad'+str(pad)
    if bi:
        modelName = modelName+'_bi'
        frame_len4dataset = frame_len*2-1
    else:
        frame_len4dataset = frame_len
    #모델과 tensorboard graph를 저장할 이름
    saveName = f'{input_type.upper()}model-{modelName}_frame-{frame_len}_sample-{sampling}_split-{splitFileName}_LR-{lr}_batch-{train_batch}_loss-{loss_type}'#_iou-{iouTH}
    log_dir = os.path.join(save_dir, folderName, saveName)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    load_whole=False
    '''
    train_dataset = conv3dVari.VFSS_data(split_file = split_file, split = 'train', root = root, frame_len=frame_len, 
        sampling=sampling, resize=resize, num_classes=num_classes, transforms=transform, input_type=input_type, load_whole=load_whole)
    '''
    print('load train data')
    train_dataset = vfssData.VFSS_data(split_file = split_file, split = 'train', rgb_root=rgb_root, flow_root=flow_root, frame_len=frame_len4dataset, 
        fill=fill, sampling=sampling, resize=resize, num_classes=num_classes, transforms=transform, load_whole=load_whole, gray_scale=gray_scale)
    if resize==(512, 512): num_workers=0
    else: num_workers=min(round(len(train_dataset)/train_batch+0.4), psutil.cpu_count()//8)
    # print('train num worker: ', num_workers)
    train_dataloader = data_utl.DataLoader(train_dataset, batch_size=train_batch, shuffle=True, num_workers=num_workers)#,worker_init_fn=fixRandom.seed_worker, generator=g)
    print('load validation data')
    val_dataset = vfssData.VFSS_data(split_file = split_file, split = 'val', rgb_root=rgb_root, flow_root=flow_root, frame_len=frame_len4dataset, 
        fill=fill, sampling=sampling, resize=resize, num_classes=num_classes, transforms=transform, load_whole=load_whole, gray_scale=gray_scale)
    if resize==(512, 512): num_workers=0
    else: num_workers=min(round(len(val_dataset)/val_batch+0.4), psutil.cpu_count()//8)
    # print('val num worker: ', num_workers)
    val_dataloader = data_utl.DataLoader(val_dataset, batch_size=val_batch, num_workers=num_workers)#,worker_init_fn=fixRandom.seed_worker, generator=g)
    
    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}

    if n_epochs_stop is not None: early_stopping = EarlyStopping(patience = n_epochs_stop, verbose = False, path=os.path.join(save_dir, folderName, saveName)+'.pth.tar')
    
    model=model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # DataParallel로 만든 state dict는 DataParallel로 만든 모델에만 넣어야 한다.
    if last_model is not None:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
    # print(summary(model, input_size=(train_batch, in_channels, frame_len, resize[0], resize[1]), verbose=0))
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print('Initial learning rate: ', optimizer.param_groups[0]['lr'])
    if loss_type=='CE':
        criterion = torch.nn.CrossEntropyLoss().to(device)  # standard crossentropy loss for classification
    elif loss_type.lower()=='custom':#labelType == 'frameLabel' or frame_len==1:
        weight = torch.tensor([1,3])# the loss weight when num_classes==2
        weight = weight/weight.sum()
        criterion = conv3dVari.PenaltyLoss(torch.nn.CrossEntropyLoss(weight=weight)).to(device)
    else:
        sys.exit('check loss type again')
    # HNS=[] #hard negative sample data list.
    
    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()
            # reset the running loss and corrects
            running_loss = 0.0
            y_true, y_score = [], []
            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':  
                model.train()
            else:
                model.eval()

            for data in tqdm(trainval_loaders[phase]):
                inputs, labels, iouLabel, patientID = data
                # move inputs and labels to the device the training is taking place on
                inputs = inputs.to(device)#Size([batch, RGB, frame_len, H, W])
                # if (modelName == 'vgg') or 'efficient' in modelName:
                #     print(inputs.size())
                #     inputs = inputs.permute(0, 2, 1, 3, 4).contiguous()
                #     inputs = torch.flatten(inputs, start_dim=0, end_dim=1)
                #     #Size([batch*frame_len, RGB, H, W])
                # elif modelName=='TDN':# make Size([batch, frame_len, RGB, W, H])
                #     inputs = inputs.permute(0, 2, 1, 4, 3).contiguous()
                optimizer.zero_grad()
                # print('totoal', inputs.size(), ', forward', inputs[:, :, :frame_len].size(), ', backward', inputs[:, :, frame_len-1:].size())
                if phase == 'train':
                    if bi:# frame_len == 5 인 경우, 00001+10000 => 000010000 이므로.
                        outputs = model(inputs[:, :, :frame_len], inputs[:, :, frame_len-1:])
                    else:
                        outputs = model(inputs)

                else:
                    with torch.no_grad():
                        if bi:
                            outputs = model(inputs[:, :, :frame_len], inputs[:, :, frame_len-1:])
                        else:
                            outputs = model(inputs)
                if torch.isnan(torch.sum(outputs)):
                    sys.exit("There is NaN in model's output")
                    
                if labelType == 'frameLabel': # outputs: torch.Size([batch, frame_len, num_classes)
                    data_len = trainval_sizes[phase] * frame_len # bi가 아닐때만 가능
                    outputs=torch.flatten(outputs, start_dim=0, end_dim=1)
                else:
                    data_len = trainval_sizes[phase] # the number of patients in a train or validation set
                    if ('i3d' in modelName) and (len(outputs.size())==3) and (outputs.size(-1)==1): 
                        outputs = outputs.squeeze(2)
                    elif modelName=='TDN': outputs = outputs.squeeze(1)
                    # print(outputs.size())#torch.Size([batch=10, num_classes=2])                  

                probs = torch.nn.Softmax(dim=1)(outputs) #torch.Size([batch=10, num_classes=2])
                preds = torch.max(probs, 1)[1] #torch.Size([frame_len])

                # if 'LSTM' in modelName:
                #     label = labels[:, 1, -1].long()
                if labelType == 'frameLabel': # elif
                    label = torch.flatten(labels[:, 1], start_dim=0).long()
                    # print('frameLabel label size:', label.size())#torch.Size([frame_len])
                elif modelName in midLabelModel: # print(labels.size())# ([batch, num_classes=2, frame_len])
                    if bi: 
                        ind = [frame_len-1] # bi 는 무조건 홀수다!! # frame_len == 5 인 경우, 00001+10000 => 000010000 이므로.
                    else:
                        if frame_len%2 == 0: # frame_len == 4 인 경우, 0110
                            ind = [int(frame_len/2-1), int(frame_len/2-2)]
                        else:# frame_len == 5 인 경우, 00100
                            ind = [int(frame_len/2)]
                        # window 중간이 positive인가?
                    # print('label index: ', ind)
                    label = (torch.sum(labels[:, 1, ind], 1)>=len(ind)).long() #size([1])
                elif labelType == 'label':
                    TH = frame_len/2
                    label = (torch.sum(labels[:, 1], 1)>=TH).long()
                    # print('3')
                elif labelType == 'IoU':
                    label = torch.as_tensor(iouLabel).long()
                    # print('4')
                
                label = label.to(device)
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

                    y_true.extend(label.tolist())
                    y_score.extend(preds.tolist())

            ##############################################################################################################
            # training or validation step finished 
            ##############################################################################################################
            epoch_loss = running_loss / data_len
            epoch_acc = metrics.accuracy_score(y_true, y_score) # running_corrects.double() / data_len
            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, num_epochs, epoch_loss, epoch_acc))
            ap = metrics.average_precision_score(y_true, y_score)#AP는 Recall에 따른 Precision 곡선 아래 면적
            #현재 클래스 하나만 detect하는 것이므로, mAP==AP

            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                if (scheduler is not None) and ('ReduceLROnPlateau' not in str(scheduler)):
                    scheduler.step()
                train_loss = epoch_loss
                train_acc = epoch_acc
                train_ap = ap
            elif phase=='val':
                if 'ReduceLROnPlateau' in str(scheduler):
                    scheduler.step(epoch_loss)
            
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

            if phase=='val':
                recall = metrics.recall_score(y_true, y_score, pos_label=1, average='binary') # TP / (TP + FN)
                precision = metrics.precision_score(y_true, y_score, pos_label=1, average='binary') # TP / TPFP
                F1_score = metrics.f1_score(y_true, y_score, pos_label=1, average='binary') # 2*precision*recall/(precision+recall)
                #FNR = FN/(FN+TP)#False Negative Rate
                fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                zero_ratio = y_score.count(0)/len(y_score) # TNFN / (data_len)
                one_ratio = y_score.count(1)/len(y_score) # TPFP / (data_len)
                print("recall: {}".format(round(recall.item(), 3)))
                print("precision: {}".format(round(precision.item(), 3)))
                print("F1_score: {}".format(round(F1_score.item(), 3)))
                print("AUC: {}".format(round(auc, 3)))
                print("AP: {}".format(round(ap, 3)))
                print("zero_ratio: {}, one_ratio: {}".format(zero_ratio, one_ratio))

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
                    # if early_stopping.counter==0: val_acc = epoch_acc
                    if early_stopping.early_stop:
                        # print("Early stopping. Saved model is from epoch ", epoch+1-n_epochs_stop)
                        # last_val_acc = round(val_acc.item(), 3)
                        model4test = os.path.join(save_dir, folderName, saveName)+'.pth.tar'
                        
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
    torch.cuda.empty_cache()
    return model4test

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
    parser.add_argument('--rgb_root', type=str, help='full path of the frames folder', default= None) # 'frames'
    #server2:/home/DATA/syj/vfss/frames #local: C:/Users/singku/Downloads/vfss/frames #server4: /DATA/jsy/vfss/frames #lab: /home/DATA/syj/frames
    parser.add_argument('--flow_root', type=str, default=None, help='full path of the flows folder')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs for training', default=100000)
    parser.add_argument('--train_batch', type=int, help='batch size', default=32)
    parser.add_argument('--val_batch', type=int, help='batch size', default=47)
    parser.add_argument('--last_model', type=str, help='last model you wanna resume', default=None)
    parser.add_argument('--num_classes', type=int, help='number of action classes including the background', default = 2)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--frame_len', type=int, default=5)#if 100, load_whole, then, batches are 1 for each, automatically
    parser.add_argument('--pad', type=int, default=None, help='fill the length with zero. 16 for resNet3D')
    parser.add_argument('--fill', type=int, default=None, help='fill the length with iteration. 16 for resNet3D')
    parser.add_argument('--bi', type=bool, default=False, help='whether train the model bidirectional. works only when model is resNet3D')
    parser.add_argument('--sampling', type=str, help='p or u or s or o', default='p')#professor's method, uniform sampling, successive sampling, make positive label one third
    parser.add_argument('--modelName', type=str, default='resNet3D')#frame_len must be 1 when the model is vgg3d or vgg
    parser.add_argument('--n_epochs_stop', type=int, help='early stopping threshold', default=50)
    parser.add_argument('--loss_type', type=str, help='custom or CE', default = 'CE')
    args = parser.parse_args()

    

    model4test = train_model(bi = args.bi, train_batch=args.train_batch, val_batch=args.val_batch, num_epochs=args.num_epochs, last_model=args.last_model, fill = args.fill, 
    frame_len = args.frame_len, sampling = args.sampling, modelName = args.modelName, split_file=args.split_file, loss_type = args.loss_type, lr=args.lr, 
    rgb_root = args.rgb_root, flow_root = args.flow_root, num_classes=args.num_classes, n_epochs_stop=args.n_epochs_stop, pad = args.pad)

    print("the saved mode's name is ", model4test)