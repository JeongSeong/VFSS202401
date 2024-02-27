import os
import sys
import torch
import conv3dVari_each as conv3dVari
#tri 없음, resNet3DKeepFrameLength 제대로 작동
def returnModel(modelName, resume_epoch, save_dir, num_classes, lr, input_type, frame_len, bi, trainVal=True):
    if input_type == 'rgb': in_channels=3
    elif input_type =='flow': in_channels=2
    parentdir = os.path.dirname(save_dir)
    transform=None
    front_rear_pad=0
    win_stride = 1 # if labelType == 'frameLabel': win_stride = frame_len
    if 'resNet3D' in modelName:# https://github.com/kenshohara/3D-ResNets-PyTorch
        labelType = 'label'
        resize=(224, 224)
        # from resNet3D import generate_model
        # seever 2: '/home/DATA/syj/vfss/C3D_VGG/r3d18_K_200ep.pth', local: 'C:/Users/singku/Downloads/vfss/C3D_VGG/r3d18_K_200ep.pth'
        if 'keepframelength' in modelName.lower():
            layer = int(modelName.lower().split('keepframelength')[-1])# [10, 18, 34, 50, 101, 152, 200]
            labelType = 'frameLabel'
            win_stride = frame_len
            from resNet3DKeepFrameLength import generate_model
            model = generate_model(layer, n_classes=700) # 숫자 없던건 layer 50이었음

            if trainVal:
                if layer == 18: 
                    pretrained = torch.load(os.path.join(save_dir,'r3d18_K_200ep.pth'), map_location=lambda storage, loc: storage)
                elif layer == 34:
                    pretrained = torch.load(os.path.join(save_dir,'r3d34_K_200ep.pth'), map_location=lambda storage, loc: storage)
                elif layer == 50:
                    pretrained = torch.load(os.path.join(save_dir,'r3d50_K_200ep.pth'), map_location=lambda storage, loc: storage)
                elif layer == 101:
                    pretrained = torch.load(os.path.join(save_dir,'r3d101_K_200ep.pth'), map_location=lambda storage, loc: storage)
                elif layer == 152:
                    pretrained = torch.load(os.path.join(save_dir,'r3d152_K_200ep.pth'), map_location=lambda storage, loc: storage)
                elif layer == 200:
                    pretrained = torch.load(os.path.join(save_dir,'r3d200_K_200ep.pth'), map_location=lambda storage, loc: storage)
                
                model.load_state_dict(pretrained["state_dict"])

            if layer > 34:
                d_model = 2048
            else:
                d_model = 512
            
            if 'transformer' in modelName.lower(): 
                # dim_feedforward=int(modelName.lower().split('keepframelength')[0].split('transformer')[-1]) 
                num_layers = int(modelName.lower().split('keepframelength')[0].split('transformer')[-1]) 
                if 'PE' in modelName: # PEresNet3DTransformer피드포워드_디멘젼keepframelength레즈넷_레이어_수
                    # PEresNet3DTransformer넘keepframelength레즈넷_레이어_수
                    model.fc = torch.nn.Identity()
                    if 'U' in modelName:
                        model = conv3dVari.PeB4Fc(backbone=model, PE=conv3dVari.PE(d_model), 
                            FC=conv3dVari.Transformer_UNet(embed_dim=d_model, num_heads=4, num_classes=num_classes))
                        target_layer = model.FC.outConv
                    else:
                        model = conv3dVari.PeB4Fc(backbone=model, PE=conv3dVari.PE(d_model), 
                            FC=conv3dVari.Transformer_3D(d_model=d_model, dim_feedforward=d_model, nhead=4, num_layers=num_layers, num_classes=num_classes))
                        target_layer = model.FC.transformer
                else: # resNet3DTransformer피드포워드_디멘젼keepframelength레즈넷_레이어_수
                    model.fc = conv3dVari.Transformer_3D(d_model=d_model, dim_feedforward=d_model, nhead=8, num_layers=num_layers, num_classes=num_classes)
                    target_layer = model.fc.transformer
            else:
                if 'PE' in modelName: # PEresNet3DKeepFrameLength레즈넷_레이어_수
                    # model.fc = torch.nn.Sequential(conv3dVari.PE(d_model), 
                    # torch.nn.Linear(d_model, num_classes))
                    model.fc = torch.nn.Identity()
                    model = conv3dVari.PeB4Fc(backbone=model, PE=conv3dVari.PE(d_model), FC=torch.nn.Linear(d_model, num_classes))
                    target_layer = model.backbone.layer4[-1]
                else: # resNet3DKeepFrameLength레즈넷_레이어_수
                    model.fc = torch.nn.Linear(d_model, num_classes)
                    target_layer = model.layer4[-1]

            # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            # scheduler = None #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer) # None            
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)    
        
        elif 'attention' in  modelName.lower(): # resNet3D레즈넷_레이어_수Attention넘헤드_수 예) resNet3D18Attention1
            num_head = modelName.split('resNet3D')[-1].lower().split('attention')
            layer = int(num_head[0])# [10, 18, 34, 50, 101, 152, 200]
            ### resNet3DKeepFrameLength 와 이하 같은 부분
            labelType = 'frameLabel'
            win_stride = frame_len
            from resNet3DKeepFrameLength import generate_model
            model = generate_model(layer, n_classes=700)
            if ('F' not in modelName) and trainVal:
                if layer == 18: 
                    pretrained = torch.load(os.path.join(save_dir,'r3d18_K_200ep.pth'), map_location=lambda storage, loc: storage)
                elif layer == 34:
                    pretrained = torch.load(os.path.join(save_dir,'r3d34_K_200ep.pth'), map_location=lambda storage, loc: storage)
                elif layer == 50:
                    pretrained = torch.load(os.path.join(save_dir,'r3d50_K_200ep.pth'), map_location=lambda storage, loc: storage)
                elif layer == 101:
                    pretrained = torch.load(os.path.join(save_dir,'r3d101_K_200ep.pth'), map_location=lambda storage, loc: storage)
                elif layer == 152:
                    pretrained = torch.load(os.path.join(save_dir,'r3d152_K_200ep.pth'), map_location=lambda storage, loc: storage)
                elif layer == 200:
                    pretrained = torch.load(os.path.join(save_dir,'r3d200_K_200ep.pth'), map_location=lambda storage, loc: storage)
                
                model.load_state_dict(pretrained["state_dict"])
            ### resNet3DKeepFrameLength 와 이상 같은 부분
            
            if layer <= 34:
                emb_size=512
                # feature shape [batch, frame_length, 512]
                # model.fc = torch.nn.Linear(512, num_classes)
            else:
                emb_size=2048
                # feature shape [batch, frame_length, 2048]
                # model.fc = torch.nn.Linear(2048, num_classes)
            if 'PE' in modelName: # PEresNet3D레즈넷_레이어_수Attention넘헤드_수 예) PEresNet3D18Attention1
                # print(modelName)
                model.fc = torch.nn.Identity()
                if 'U' in modelName: # PEUresNet3D레즈넷_레이어_수Attention넘헤드_수 예) PEUresNet3D18Attention1
                    model = conv3dVari.PeB4Fc(backbone=model, PE=conv3dVari.PE(emb_size), # [batch, frameLength, embed_dim] upsample할거라 bilinear 안됨
                        FC=conv3dVari.MHA_UNet(embed_dim=emb_size, num_heads=int(num_head[-1]), num_classes=num_classes, bilinear=False))
                    target_layer = model.FC.outConv
                else:
                    model = conv3dVari.PeB4Fc(backbone=model, PE=conv3dVari.PE(emb_size), 
                        FC=conv3dVari.MHA_3D(embed_dim=emb_size, num_heads=int(num_head[-1]), num_classes=num_classes))
                    target_layer = model.backbone.layer4[-1]
                    if 'F' in modelName:
                        if not trainVal: # if trainVal is False # str(type(trainVal))!="<class 'bool'>"
                            pretrained = torch.load(trainVal, map_location=lambda storage, loc: storage)
                            model.load_state_dict(pretrained["state_dict"])
                        model = conv3dVari.PeB4Fc(backbone=model.backbone, PE=model.PE,
                            FC=conv3dVari.MHA_UNet(embed_dim=emb_size, num_heads=int(num_head[-1]), num_classes=num_classes, bilinear=False),
                                freeze=True)
                        target_layer = model.FC.outConv
            else:
                model.fc = conv3dVari.MHA_3D(embed_dim=emb_size, num_heads=int(num_head[-1]), num_classes=num_classes)
                target_layer =  model.layer4[-1]
            # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            # scheduler = None #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer) # None            
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)    
        
        elif 'LSTM' in modelName: # 'resNet3DLSTM'
            labelType = 'frameLabel'
            layer=18
            feature_dim = 512
            from resNet3DKeepFrameLength import generate_model
            model = generate_model(layer, n_classes=700)
            pretrained = torch.load(os.path.join(save_dir,'r3d18_K_200ep.pth'), map_location=lambda storage, loc: storage)
            model.load_state_dict(pretrained["state_dict"])
            model.fc = conv3dVari.LSTM(feature_dim, num_classes)
            target_layer = model.layer4[-1] # VariTest_gradCAM.py 섞기
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            # front_rear_pad = int(frame_len-1)
            # win_stride = 1
            win_stride = frame_len
            

        else:
            from resNet3D import generate_model
            layer = modelName.split('3D')[-1]
            if layer == '': layer=18
            else: layer = int(layer)

            if layer <= 34: feature_dim = 512
            elif layer >= 50: feature_dim = 2048
            
            if trainVal: 
                if layer == 18:
                    pretrained = torch.load(os.path.join(save_dir,'r3d18_K_200ep.pth'), map_location=lambda storage, loc: storage)
                elif layer == 34:
                    pretrained = torch.load(os.path.join(save_dir,'r3d34_K_200ep.pth'), map_location=lambda storage, loc: storage)
                elif layer == 50:
                    pretrained = torch.load(os.path.join(save_dir,'r3d50_K_200ep.pth'), map_location=lambda storage, loc: storage)
                elif layer == 101:
                    pretrained = torch.load(os.path.join(save_dir,'r3d101_K_200ep.pth'), map_location=lambda storage, loc: storage)
                elif layer == 152:
                    if trainVal: pretrained = torch.load(os.path.join(save_dir,'r3d152_K_200ep.pth'), map_location=lambda storage, loc: storage)
                elif layer == 200:
                    if trainVal: pretrained = torch.load(os.path.join(save_dir,'r3d200_K_200ep.pth'), map_location=lambda storage, loc: storage)
                else: sys.exit('there is no pretrained weights for this layer')
        
            if bi:
                forward_model = generate_model(layer, n_classes=700)
                backward_model = generate_model(layer, n_classes=700)
                if trainVal: 
                    forward_model.load_state_dict(pretrained["state_dict"])
                    backward_model.load_state_dict(pretrained["state_dict"])
                forward_model.fc = torch.nn.Identity() # feature dim 512
                backward_model.fc = torch.nn.Identity() # feature dim 512
                model = conv3dVari.Both1Lin(forward_model, backward_model, feature_dim*2, num_classes)
                target_layer =  model.forward_model.layer4[-1] #, model.backward_model.layer4[-1]] # VariTest_gradCAM.py 섞기
                front_rear_pad = frame_len-1 # frame_len == 5 인 경우, 00001+10000 => 000010000 이므로.
                win_stride = 1
            else:
                model = generate_model(layer, n_classes=700)
                if trainVal: 
                    if (layer != 50) or ('pt' in modelName):
                        print('using pretrained model')
                        model.load_state_dict(pretrained["state_dict"])
                model.fc = torch.nn.Linear(feature_dim, num_classes)
                target_layer =  model.layer4[-1] # VariTest_gradCAM.py 섞기
                if frame_len%2==0:# frame_len == 4 인 경우, 0110 이므로.
                    # sys.exit('only odd numbers are available')
                    front_rear_pad = int(frame_len/2-1)
                    win_stride = 2
                else:# frame_len == 3 인 경우, 010 이므로.
                    front_rear_pad = int(frame_len/2)
                    win_stride = 1

            if layer == 50:
                # labelType = 'IoU'
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer) # 똑같이 해도 안될듯
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
                # 아래 optimizer나 schedular에서는 resNet3D 34나50이 좋은거 같음
                # 11 frame 씩 학습 시 PRT, PTT, OPD는 101 layer가 좋음 (> 0.8)
                # 11 frame 씩 학습 시 LVCDuration은 34나 50 layer가 괜찮음 (> 0.7)
                
                # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
                # scheduler = None 

    elif 'i3d' in modelName:# https://github.com/piergiaj/pytorch-i3d
        labelType = 'label'
        resize=(224, 224)
        if not os.path.exists(os.path.join(save_dir, 'pytorch-i3d')):
            sys.path.insert(0, os.path.join(parentdir, 'pytorch-i3d'))   
        else: sys.path.insert(0, os.path.join(save_dir, 'pytorch-i3d'))
        from pytorch_i3d import InceptionI3d
        if trainVal:
            if input_type=='rgb':
                pretrained_weight = 'models/rgb_imagenet.pt'
            elif input_type=='flow':
                pretrained_weight = 'models/flow_imagenet.pt'
            pretrained = os.path.join(save_dir, 'pytorch-i3d', pretrained_weight)
            if not os.path.isfile(pretrained): pretrained  = os.path.join(parentdir, 'pytorch-i3d', pretrained_weight)
        if bi:
            forward_model = InceptionI3d(400, in_channels=in_channels)
            backward_model = InceptionI3d(400, in_channels=in_channels)
            if trainVal:
                forward_model.load_state_dict(torch.load(pretrained))
                backward_model.load_state_dict(torch.load(pretrained))#torch.Size([32, 1024, 1])
            forward_model.logits = conv3dVari.Identity()
            backward_model.logits = conv3dVari.Identity()
            model = conv3dVari.Both1Lin(forward_model, backward_model, 1024*2, num_classes)
            target_layer =  model.forward_model._modules['Mixed_5c'] # VariTest_gradCAM.py 섞기
            front_rear_pad = frame_len-1 # frame_len == 5 인 경우, 00001+10000 => 000010000 이므로.
            win_stride = 1
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        else:
            model = InceptionI3d(400, in_channels=in_channels)
            if trainVal: model.load_state_dict(torch.load(pretrained))
            model.replace_logits(num_classes)
            target_layer =  model._modules['Mixed_5c'] # VariTest_gradCAM.py 섞기
            if frame_len%2==0:# frame_len == 10 인 경우, 0000110000 이므로.
                sys.exit('only odd frame numbers are available')
                front_rear_pad = int(frame_len/2-1)
                win_stride = 2
            else:# frame_len == 9 인 경우, 000010000 이므로.
                front_rear_pad = int(frame_len/2)
                win_stride = 1
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    elif 'vgg' in modelName:
        resize = (224, 224)#shape (3 x H x W)
        labelType = 'label'
        import torchvision.models as models
        from torchvision import transforms
        #range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        transform=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if trainVal: pretrained=True
        else: pretrained=False
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=2)
        win_stride = 1
        target_layer =  model.features[-1] # VariTest_gradCAM.py 섞기

        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)


    if 'pretrained' in locals(): del pretrained

    return resize, labelType, model, optimizer, scheduler, transform, front_rear_pad, win_stride, target_layer