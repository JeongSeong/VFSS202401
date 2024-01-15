import os
import sys

def findFile(file, save_dir):
    fileName = file.split('/')[-1]
    if not os.path.isfile(file):
        file = os.path.join(save_dir, fileName)
        if not os.path.isfile(file):
            parentdir = os.path.dirname(save_dir)
            file = os.path.join(parentdir, fileName)
            if not os.path.isfile(file): 
                sys.exit(f'specify the whole path of the {fileName} file')
    fileName = fileName.split('.')[0]
    return file, fileName

def findDir(root, save_dir):
    if root is not None:
        if not os.path.exists(root):
            name = root.split('/')[-1]
            root = os.path.join(save_dir, name)
            if not os.path.exists(root):
                parentdir = os.path.dirname(save_dir)
                root = os.path.join(parentdir, name)
                if not os.path.exists(root):
                    sys.exit(f'specify the whole path of the {name} directory')
    return root
