import subprocess
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--program', default='vfssRGBsingleTrainTest_fixed.py', type=str)
parser.add_argument('--train_type', default='normal', type=str, help='or _variMem')
parser.add_argument('--win_limit', type=int, default=50, #50, 
    help='maximum frame batch that can be loaded on a GPU. resNet3D layer:win_limit = 34:60, 50:50, 101:20 152: 200:')
parser.add_argument('--input_type', default='rgb', type=str, help='flow, rgb, both')
parser.add_argument('--suffix', default='_withSkipped', type=str, help='FinStrokeTest, FinStrokeTest-interpolated, FinStrokeTestWithSkipped, _interpolated, _withSkipped, withSkippedBalanced')
parser.add_argument('--startLR', default=1, type=int, help='will be divided depend on model')
parser.add_argument('--endLR', default=10, type=int, help='will be divided depend on model')
parser.add_argument('--variLr', default=False, type=bool)
parser.add_argument('--exponent', default=3, type=int)
parser.add_argument('--includeEvent', default=None, type=str, nargs='+', help='separate with space bar')
parser.add_argument('--exceptEvent', default=None, type=str, nargs='+', help='separate with space bar')
parser.add_argument('--repeat', default=20, type=int, help='repeatition for result averaging')
# parser.add_argument('--variEvent', default=True, type=bool)
########################################################################################################
parser.add_argument('--modelName', default='resNet3D', type=str, help='i3d, vgg')
parser.add_argument('--frame_len', type=int, default=7)
parser.add_argument('--fill', type=int, default=None, help='fill the length with iteration. 16 for resNet3D')
parser.add_argument('--bi', type=bool, default=False, help='whether train the model bidirectional. works only when model is resNet3D')
parser.add_argument('--sampling', type=str, help='p or u or s or o', default='p')
parser.add_argument('--train_batch', type=int, help='batch size', default=32)
parser.add_argument('--val_batch', type=int, help='batch size', default=49)
parser.add_argument('--n_epochs_stop', type=int, help='early stopping threshold', default=50) #예전에는 50으로 했었음
# parser.add_argument('--num_epochs', type=int, help='Number of epochs for training', default=100000)
parser.add_argument('--loss_type', type=str, help='custom or CE', default = 'CE')
# parser.add_argument('--num_classes', type=int, help='number of action classes including the background', default = 2)
# ########################################################################################################
parser.add_argument('--win_stride', type=int, help='window sliding stride. at most frame_len', default=None)
# parser.add_argument('--output_type', type=str, help='cluster or oneHot', default='oneHot')
args=parser.parse_args()

# if 'resNet3D' in args.modelName:
#     if 'keepframelength' in args.modelName.lower():
#         if '101' in args.modelName:
#             lr=list(map(lambda x: x/10**5, range(args.startLR, args.endLR+1)))
#         elif '18' in args.modelName:
#             lr=list(map(lambda x: x/10**4, range(args.startLR, args.endLR+1)))
#         elif args.frame_len == 16: 
#             lr=list(map(lambda x: x/10**4, range(args.startLR, args.endLR+1)))
#         else: # 10은 이게 좋았음
#             lr=list(map(lambda x: x/10**5, range(args.startLR, args.endLR+1)))
#     else:
#         if args.frame_len == 11:
#             lr=list(map(lambda x: x/10**3, range(args.startLR, args.endLR+1)))
#         else:
#             lr=list(map(lambda x: x/10**5, range(args.startLR, args.endLR+1)))
# elif args.modelName=='i3d':
#     lr=list(map(lambda x: x/100, range(args.startLR, args.endLR+1)))
# elif args.modelName=='vgg':
#     lr=list(map(lambda x: x/10**6, range(args.startLR, args.endLR+1)))
if args.variLr:
    print('learning rate variation mode')
    lr = list(map(lambda x: x/10**args.exponent, range(args.startLR, args.endLR+1)))
else: 
    print('fixed learning rate mode')
    lr=[5/10**args.exponent]
print(lr)
if args.includeEvent is not None:
    events=args.includeEvent
else: events=['start2bpm', 'bpm2hyoid', 'hyoid2uesClose', 'bpm2uesClose', 'hyoid2lvc', 'lvc2lvcOff', 'uesOpen2uesClose']
if args.exceptEvent is not None: 
    for e in args.exceptEvent:
        events.remove(e)

command=['python', f'{args.program}', '--sampling', f'{args.sampling}', '--loss_type', f'{args.loss_type}', 
'--modelName', f'{args.modelName}', '--frame_len', f'{args.frame_len}', '--n_epochs_stop', f'{args.n_epochs_stop}', 
'--train_batch', f'{args.train_batch}', '--val_batch', f'{args.val_batch}', '--win_stride', f'{args.win_stride}']

if args.fill is not None: command.extend(['--fill', f'{args.fill}'])
elif args.bi: command.extend(['--bi', 'True'])

if args.input_type == 'rgb':
    if 'inter' in args.suffix: command.extend(['--rgb_root','frames_inter'])
    else: command.extend(['--rgb_root','frames'])
elif args.input_type == 'flow':
    if 'inter' in args.suffix: command.extend(['--flow_root','flows_inter'])
    else: command.extend(['--flow_root','flows'])
else:
    import sys
    sys.exit('not supported yet')

if 'TrainTest' in args.program:
    command.extend(['--train_type', f'{args.train_type}'])
if args.win_limit is not None:
    command.extend(['--win_limit', f'{args.win_limit}'])

if args.suffix[0] != '_':
    suffix = '-'+args.suffix
else:
    suffix = args.suffix

for e in events:
    command1=command[:]
    command1.extend(['--split_file', f'{e+suffix}.xlsx'])
    print(command1)
    for r in lr:
        for _ in range(args.repeat):
            command2=command1[:]
            command2.extend(['--lr', f'{r}'])
            subprocess.run(command2)
        