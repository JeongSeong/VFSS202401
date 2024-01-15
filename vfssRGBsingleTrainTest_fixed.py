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

conda install scikit-image

conda install -c conda-forge torchinfo
'''

if __name__ == '__main__':
    # os.path.dirname(os.path.abspath(__file__)) # 현재 파일이 있는 디렉토리
    # os.path.join(os.path.dirname(os.path.abspath(__file__)), '../frames')
    # os.path exists
    # os.isfile
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_file', type=str, help='split file name', default='bpm2uesClose_withSkipped.xlsx')
    #server2:/home/DATA/syj/vfss/C3D_VGG/bpm2ues_close_out.xlsx #local: C:/Users/singku/Downloads/vfss/C3D_VGG/bpm2ues_close_out.xlsx 
    # #server4: /DATA/jsy/vfss/bpm2ues_close_out.xlsx #lab: /home/DATA/syj/vfss/bpm2ues_close_out.xlsx
    parser.add_argument('--rgb_root', type=str, help='full path of the frames folder', default= None) #'frames_inter'
    #server2:/home/DATA/syj/vfss/frames #local: C:/Users/singku/Downloads/vfss/frames #server4: /DATA/jsy/vfss/frames #lab: /home/DATA/syj/frames
    parser.add_argument('--flow_root', type=str, default=None, help='full path of the flows folder')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs for training', default=100000)
    parser.add_argument('--train_batch', type=int, help='batch size', default=32)
    parser.add_argument('--val_batch', type=int, help='batch size', default=49)
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
    parser.add_argument('--train_type', type=str, default='normal', help='or _variMem')
    parser.add_argument('--win_limit', type=int, default=50, help='maximum frame batch that can be loaded on a GPU')
    parser.add_argument('--trainVal4F', type=str, help='pretrained weight full path', default=True)
    ########################################################################################################
    parser.add_argument('--win_stride', type=int, help='window sliding stride. at most frame_len', default=None)
    # parser.add_argument('--output_type', type=str, help='cluster or oneHot', default='oneHot')
    args = parser.parse_args()

    if args.train_type == 'normal': 
        if 'PE' in args.sampling: 
            import sys
            sys.exit('PE sampling mode is only for _variMem train_type')
        from vfssRGBsingleTrain_fixed import train_model
        model4test = train_model(bi = args.bi, train_batch=args.train_batch, val_batch=args.val_batch, num_epochs=args.num_epochs, last_model=args.last_model, fill = args.fill, 
            frame_len = args.frame_len, sampling = args.sampling, modelName = args.modelName, split_file=args.split_file, loss_type = args.loss_type, lr=args.lr, 
            rgb_root = args.rgb_root, flow_root = args.flow_root, num_classes=args.num_classes, n_epochs_stop=args.n_epochs_stop, pad = args.pad)
    elif args.train_type.lower() == '_varimem': 
        if ('transformer' in args.modelName.lower()) or ('attention' in args.modelName.lower()) or ('keep' in args.modelName.lower()):
            from vfssRGBsingleTrain_valoadWhole import train_model
            model4test = train_model(bi = args.bi, train_batch=args.train_batch, val_batch=args.val_batch, num_epochs=args.num_epochs, last_model=args.last_model, fill = args.fill, 
            frame_len = args.frame_len, sampling = args.sampling, modelName = args.modelName, split_file=args.split_file, loss_type = args.loss_type, lr=args.lr, 
            rgb_root = args.rgb_root, flow_root = args.flow_root, num_classes=args.num_classes, n_epochs_stop=args.n_epochs_stop, pad = args.pad, trainVal4F=args.trainVal4F)
        else: 
            from vfssRGBsingleTrain_variMem import train_model # frameLabel 작동 못함. vfssTransformer.py도 frameLabel 작동 못함
            model4test = train_model(bi = args.bi, train_batch=1, val_batch=1, num_epochs=args.num_epochs, last_model=args.last_model, fill = args.fill, 
            frame_len = args.frame_len, sampling = 'loadWhole', modelName = args.modelName, split_file=args.split_file, loss_type = args.loss_type, lr=args.lr, 
            rgb_root = args.rgb_root, flow_root = args.flow_root, num_classes=args.num_classes, n_epochs_stop=args.n_epochs_stop, pad = args.pad, win_limit=args.win_limit)
        
    if 'PE' in args.sampling: 
        # from vfssRGBsingleTest_variMem_PE1stF import test_model
        from vfssRGBsingleTest_variMem_overlap import test_model
        # test_model(last_model=model4test, rgb_root = args.rgb_root, flow_root = args.flow_root, split_file=args.split_file, 
        #     num_classes=args.num_classes, win_limit=args.win_limit)# win_stride=args.win_stride, 
        test_model(last_model=model4test, rgb_root = args.rgb_root, flow_root = args.flow_root, split_file=args.split_file, 
            num_classes=args.num_classes, win_limit=args.win_limit, win_stride=args.win_stride)# , 
    else:
        # last_model_split = model4test.split('/') #last_model은 항상 모델이 있는 폴더명부터 줘야 함
        # if ('frameL' in last_model_split[-2]) and (args.win_stride < args.frame_len): # 이거 에러나니까 쓰지 마라
        #     from vfssRGBsingleTest import test_model # stride를 frame_len보다 작게 해서 overlap해서 and 연산 가능
        #     test_model(last_model=model4test, rgb_root = args.rgb_root, flow_root = args.flow_root, split_file=args.split_file, 
        #         win_stride=args.win_stride, num_classes=args.num_classes, output_type=args.output_type)
        # else:
        #     # else:
        #     #     from vfssRGBsingleTestBatchSize import test_model
        #     #     test_model(last_model=model4test, rgb_root = args.rgb_root, flow_root = args.flow_root, split_file=args.split_file, 
        #     #         win_stride=args.win_stride, num_classes=args.num_classes, output_type=args.output_type)
        
        from vfssRGBsingleTest_variMem import test_model
        test_model(last_model=model4test, rgb_root = args.rgb_root, flow_root = args.flow_root, split_file=args.split_file, 
            num_classes=args.num_classes, win_limit=args.win_limit)# win_stride=args.win_stride, 