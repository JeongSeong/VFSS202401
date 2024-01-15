##########################################
# code to generate RGB & flow frames from video
# finding missing frames,
# create a datafile in csv format
##########################################

'''
conda create --name syj_vfss_prepro python=3.7
pip install pytesseract
conda install openpyxl
conda install pandas
conda install -c conda-forge opencv

installed tesseract-ocr

put snum.traineddata in 
/usr/share/tesseract-ocr/4.00/tessdata
참고로 snum.traineddata는 Dropbox (Korea Univ. Department of Computer)/SYC 에 있음. 잘 찾아보길.

그리고 open-cv는 
pip install opencv-python
으로 검색해서 설치하기!!
'''
import argparse
import cv2
import os
import os.path
import pandas as pd
import pytesseract


parser=argparse.ArgumentParser()
parser.add_argument('--vid_root', type=str, help='clips directory', default='/data/syj/vfss/clips')
parser.add_argument('--rgb_root', type=str, help='frames directory', default='/data/syj/vfss/frames')
parser.add_argument('--flow_root', type=str, help='flows directory', default='/data/syj/vfss/flows')
parser.add_argument('--annotation', type=str, help='location of annotation.xlsx', default='/data/syj/vfss/annotation.xlsx')
parser.add_argument('--tesseract', type=str, help='tesseract directory', default=r'/usr/bin/tesseract')
# desktop directory: "C:/Program Files/Tesseract-OCR/tesseract"
args=parser.parse_args()
pytesseract.pytesseract.tesseract_cmd = args.tesseract
videoroot=args.vid_root
imgroot=args.rgb_root
flowroot=args.flow_root
# create directory if does not exist
try: os.mkdir(imgroot)
except OSError as error: print('use existing frames dir')
try: os.mkdir(flowroot)
except OSError as error: print('use existing flows dir')
# the abbreviations of the column names
annotate_cols = {'sn':0,
                 'start': 1,
                 'bpm': 2,
                 'hyoid': 3,
                 'laryngeal': 4,
                 'lvc': 5,
                 'ues open': 6,
                 'mpc': 7,
                 'ues close': 8,
                 'lvc off': 9,
                 'swallow rest': 10}
# read the annotation excel file
annot_df = pd.read_excel(args.annotation)

# save some columns (serial number, ues opening and closure) in the variables
sn_key = annot_df.columns[annotate_cols['sn']]
lvc_key = annot_df.columns[annotate_cols['bpm']]
lvcoff_key = annot_df.columns[annotate_cols['ues close']]

# make the serial number as an index of annot_df
annot_df.set_index(sn_key, drop=False, inplace=True)

'''
# to get ues opening duration except infinity or nan values.
lvc = annot_df[lvc_key].values
lvcoff = annot_df[lvcoff_key].values
lvc = lvc[np.isfinite(lvc)]
lvcoff = lvcoff[np.isfinite(lvcoff)]
lvc_dur = lvcoff-lvc 
'''
# create dataframe
columns = ['patient_id', 'start_frame', 'end_frame', 'num_frame', 'label_start', 'label_end', 'split']
df = pd.DataFrame(columns=columns)

video_cnt = 0

L=-20
H=20
xx1 = 128
xx2 = 1151
minimum=255 #a variable that save the minimum pixel of the frames of whoel videos.
maximum=0 #a variable that save the maximum pixel of the frames of whoel videos.
for vname in os.listdir(videoroot):
    print('\nfile name: ', vname)
    # get patient ID
    patientId = vname.split('_')[0].zfill(4)

    # create directory if does not exist
    patient_frame_dir=os.path.join(imgroot,patientId)
    patient_flow_dir=os.path.join(flowroot,patientId)
    try: os.mkdir(patient_frame_dir)
    except OSError as error: print("use existing  patient's frames dir")
    try: os.mkdir(patient_flow_dir)
    except OSError as error: print("use existing  patient's flows dir")

    # save frames?
    framecnt = 0
    first_frame=0
    last_frame=0
    cur_frame =-1
    # f3d = [] # print('frame 3d', np.array(f3d).shape)
    # extract_fnum = True
    missing=[]
    # scale_percent = 25 # percent of original size
    cap = cv2.VideoCapture(os.path.join(videoroot, vname))
    while(cap.isOpened()):
        # 비디오의 한 프레임씩 읽습니다. 
        # 제대로 프레임을 읽으면 ret값이 True, 실패하면 False.
        # frame에 읽은 프레임이 나옵니다
        ret, frame = cap.read()

        if frame is None: break

        # convert bgr image to grayscale
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        # extract frame number region
        img = gframe[0:28,1224:1280]
        # black <-> white 
        img=cv2.bitwise_not(img)
        # OCR
        text2 = pytesseract.image_to_string(image=img, lang='snum', config='--oem 3 --psm 6 outputbase digits')
        try: fno = int(text2) # frame number
        except:
            print('no frame number detected! continuing..')
            continue

        # black <-> white frame
        frame =cv2.bitwise_not(frame)
        # save the first frame number
        if framecnt==0:
            first_frame = fno
            print('first frame number: ',first_frame)

        # if cur_frame >= fno, it means there are identical frames
        if cur_frame < fno:
            # if it is not the first frame and not the next frame
            if fno != cur_frame+1 and cur_frame != -1:
                print('#############################################')
                print(f'missed frame! missing number: {cur_frame+1}')
                missing.append(cur_frame+1)
            # put the frame numer as the current frame
            cur_frame = fno 
            # if a missing frame number is found, dlete it from the missing list
            if cur_frame+1 in missing: missing.remove(cur_frame+1)
            # save the frame
            fname = patientId+'_'+str(cur_frame).zfill(5)+'.jpg'
            frame2save=frame[:,xx1:xx2+1,:]
            frameMin=frame2save.min()
            frameMax=frame2save.max()
            # print(frameMin, frameMax)
            if minimum > frameMin:
                minimum = frameMin
            if maximum < frameMax:
                maximum = frameMax
            cv2.imwrite(os.path.join(patient_frame_dir,fname),frame2save)

            # Get the optical flow
            cur_f = gframe[:,xx1:xx2+1]
            if framecnt > 0:  # start from frame 1
                flow = cv2.calcOpticalFlowFarneback(prv_f, cur_f, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
                fx = flow[...,0]
                fy = flow[...,1]
            
                nfx = 255*(fx-L)/(H-L)
                nfx[nfx<0]=0
                nfx[nfx>255]=255

                nfy = 255*(fy-L)/(H-L)
                nfy[nfy<0]=0
                nfy[nfy>255]=255

                fnamex = patientId+'_'+str(cur_frame).zfill(5)+'x.jpg'
                fnamey = patientId+'_'+str(cur_frame).zfill(5)+'y.jpg'

                cv2.imwrite(os.path.join(patient_flow_dir,fnamex),nfx)
                cv2.imwrite(os.path.join(patient_flow_dir,fnamey),nfy)

            # set previous frame
            prv_f = cur_f
            
        framecnt += 1
    '''
    print(f'there are missing frames {missing} for patient ID: {patientId}') 
    last_frame = cur_frame
    actual_fcnt = last_frame - first_frame + 1 - len(missing)
    print('frame count', framecnt)
    print('first frame', first_frame)
    print('last frame', last_frame)
    print('actual frame count', actual_fcnt)

    sn=int(patientId)
    if missing==[] and sn in annot_df[sn_key].values:
        split='train' if sn <= 100 else 'test'
        lvc_start = annot_df.loc[sn, lvc_key]
        lvc_end = annot_df.loc[sn, lvcoff_key]
        # adding a row
        df.loc[sn] = [sn, first_frame, last_frame, actual_fcnt, lvc_start, lvc_end, split]  
    '''
    cap.release() # 오픈한 캡쳐 객체를 해제합니다
    #cv2.destroyAllWindows()
    
    video_cnt += 1
    print('video count:', video_cnt)
'''
df.sort_index(inplace=True)
print(df)
df.to_excel("PTTout.xlsx")
'''