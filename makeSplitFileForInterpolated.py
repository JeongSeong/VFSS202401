#make balanced data index output file for the UES opening duration localization

'''
conda create --name syj_vfss_prepro python=3.7
#pip install pytesseract
conda install openpyxl
conda install pandas
#conda install -c conda-forge opencv

#installed tesseract-ocr

#snum.traineddata is in 
#/usr/share/tesseract-ocr/4.00/tessdata
'''

import os
import os.path
import argparse
import pandas as pd
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--SL', type=str, help='starting label to detect', default='bpm')
parser.add_argument('--EL', type=str, help='ending label to detect', default='uesClose')
parser.add_argument('--annotation', type=str, help='annotation excel file root', default='annotation.xlsx')#server2: /home/DATA/syj/vfss/annotation.xlsx 
#local: C:/Users/singku/Downloads/vfss/annotation.xlsx #lab: /home/DATA/syj/vfss/annotation.xlsx #server4: /DATA/jsy/vfss/annotation.xlsx #server1: /DATA/jsy/vfss/annotation.xlsx
#server2 stroke:/home/DATA/syj/vfss/C3D_VGG/stroke_annotations.xlsx
parser.add_argument('--imgroot', type=str, help='root of frames', default='new_frames_inter')#server2: /home/DATA/syj/vfss/frames
#local: C:/Users/singku/Downloads/vfss/frames #lab: /home/DATA/syj/frames #server4: /DATA/jsy/vfss/frames #server1: /DATA/jsy/vfss/frames
parser.add_argument('--last_data', type=int, help='the last patient number you want to train', default=None)# 269, 322, 379, 431, 483, 537, 590
args=parser.parse_args()

annotate_cols = {'sn':0,
                 'start': 1,
                 'bpm': 2,
                 'hyoid': 3,
                 'lvc': 4,
                 'uesOpen': 5,
                 'mpc': 6,
                 'uesClose': 7,
                 'lvcOff': 8,
                 'SR': 9}# swallow-rest
save_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 파일이 있는 디렉토리
parentdir = os.path.dirname(save_dir)

annot_df = args.annotation
if not os.path.isfile(annot_df):
    Name = annot_df.split('/')[-1]
    annot_df = os.path.join(save_dir, Name)
    if not os.path.isfile(annot_df):
        annot_df = os.path.join(parentdir, Name)
        if not os.path.isfile(annot_df): sys.exit('specify the annotation excel file root')
# read the annotation excel file
annot_df = pd.read_excel(annot_df, engine='openpyxl')
sn_key = annot_df.columns[annotate_cols['sn']]
annot_df.set_index(sn_key, drop=False, inplace=True)
SL_key = annot_df.columns[annotate_cols[args.SL]]
EL_key = annot_df.columns[annotate_cols[args.EL]]
anno_id = annot_df[sn_key].values
columns = ['patient_id', 'num_frame', 'label_start', 'label_end', 'split']
df = pd.DataFrame(columns=columns)

rgb_root = args.imgroot
if not os.path.exists(rgb_root):
    Name = rgb_root.split('/')[-1]
    rgb_root = os.path.join(save_dir, Name)
    if not os.path.exists(rgb_root):
        rgb_root = os.path.join(parentdir, Name)
        if not os.path.exists(rgb_root): sys.exit('specify the rgb_root directory')

for dname in sorted(os.listdir(rgb_root)):
    if (args.last_data is not None) and (int(dname)>args.last_data): continue
    if int(dname) not in anno_id: 
        print('annotation is missing for patient ID: ', dname)
        continue

    missing = False
    # print(dname)
    lvc_start = annot_df.loc[int(dname), SL_key]
    lvc_end = annot_df.loc[int(dname), EL_key]
    lvc_start, lvc_end = int(min(lvc_start, lvc_end)), int(max(lvc_start, lvc_end))
    # print(lvc_start, lvc_end)
    fnameList = os.listdir(os.path.join(rgb_root, dname))
    # print(fnameList)
    fnameList = sorted(list(map(lambda x: x.split('_')[-1].split('.')[0], fnameList)))
    # print(fnameList)
    # first_frame = fnameList[0] # int(fnameList[0].split('_')[-1].split('.')[0])
    # last_frame = fnameList[-1] # int(fnameList[-1].split('_')[-1].split('.')[0])
    num_frame = len(fnameList)
    SL_count = 0
    while True:
        SL_count+=1
        SL=str(lvc_start).zfill(5)
        if SL in fnameList:
            SL = fnameList.index(SL)
            break
        elif SL_count > num_frame:
            print(f"can't find starting label's frame in patient {dname}'s video")
            missing = True
            break
        else:
            lvc_start = lvc_start+1
    EL_count = 0
    while True:
        EL_count+=1
        EL=str(lvc_end).zfill(5)
        if EL in fnameList:
            EL = fnameList.index(EL)
            break
        elif EL_count > num_frame:
            print(f"can't find ending label's frame in patient {dname}'s video")
            missing = True
            break
        else:
            lvc_end = lvc_end-1
    if missing: continue
    df.loc[-1] = [dname, num_frame, SL, EL, None]  # adding a row
    df.index = df.index + 1  # shifting index

# df.sort_index(inplace=True)
df['patient_id'] = pd.to_numeric(df['patient_id'], downcast='integer')
df.set_index('patient_id', drop=False, inplace=True)

if args.last_data == None:
    tot_data_len = len(df)
    test_data_len = int(tot_data_len*0.1)
    trainVal_data_len = tot_data_len - test_data_len
    val_data_len = int(trainVal_data_len*0.1)
    train_data_len = trainVal_data_len - val_data_len
    print(f"train:{train_data_len}, validation:{val_data_len}, test:{test_data_len}")
else:
    tot_data_len = len(df[df.index<=args.last_data])
    test_data_len = int(tot_data_len*0.1)
    trainVal_data_len = tot_data_len - test_data_len
    val_data_len = int(trainVal_data_len*0.1)
    train_data_len = trainVal_data_len - val_data_len
    print(f"train:{train_data_len}, validation:{val_data_len}, test:{test_data_len}")

len_count=0

for idx in df['patient_id'].values:
    # lvc_start = annot_df.loc[idx, SL_key]
    # lvc_end = annot_df.loc[idx, EL_key]
    # df.loc[idx, 'label_start'] = min(lvc_start, lvc_end)
    # df.loc[idx, 'label_end'] = max(lvc_start, lvc_end)
    len_count+=1

    if len_count <= train_data_len:
        split = 'train'
    elif len_count <= trainVal_data_len:
        split = 'val'
    elif len_count <= tot_data_len:
        split = 'test'
    else:
        sys.exit('there must be something wrong')

    df.loc[idx, 'split'] = split

if 'stroke' in args.annotation.split('/')[-1]:
    df.to_excel(f'stroke-{args.SL}2{args.EL}_interpolated.xlsx')
elif args.last_data != None:
    df.to_excel(f'data{args.last_data}-{args.SL}2{args.EL}_interpolated.xlsx')
else:
    df.to_excel(f'{args.SL}2{args.EL}_interpolated.xlsx')