import os
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

path = osp.join(this_dir, 'networks')
add_path(path)

path = osp.join(this_dir, 'utils')
add_path(path)

path = osp.join(this_dir, 'criteria')
add_path(path)

path = osp.join(this_dir, 'model')
add_path(path)

path = osp.join(this_dir, 'dataset')
add_path(path)

path = osp.join(this_dir, 'options')
add_path(path)

os.getcwd()