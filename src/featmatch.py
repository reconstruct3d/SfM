import cv2 
import numpy as np 
import pickle as pkl 
import argparse
import os 

from utils import * 

def FeatMatch(opts): 
    img_names = sorted(os.listdir(opts.data_dir))
    img_paths = [os.path.join(opts.data_dir, x) for x in img_names]
    
    if not os.path.exists(os.path.join(opts.out_dir,'features')): 
        os.makedirs(os.path.join(opts.out_dir,'features'))

    if not os.path.exists(os.path.join(opts.out_dir,'matches')): 
        os.makedirs(os.path.join(opts.out_dir,'matches'))

    for img_path in img_paths: 
        img = cv2.imread(img_path)
        img = img[:,:,::-1]
        feat = getattr(cv2.xfeatures2d, '{}_create'.format(opts.features))()
        kp, desc = feat.detectAndCompute(img,None)
        

def SetArguments(parser): 

    #directories stuff
    parser.add_argument('--data_dir',action='store',type=str,default='../data/fountain-P11/images/',
                        dest='data_dir') 
    parser.add_argument('--ext',action='store',type=str,default='jpg,png',dest='ext') 
    parser.add_argument('--out_dir',action='store',type=str,default='../data/fountain-P11/',
                        dest='out_dir') 

    #feature matching args
    parser.add_argument('--features',action='store', type=str, default='SURF', dest='features') 
    parser.add_argument('--matcher',action='store', type=str, default='brute-force', dest='matcher') 
    parser.add_argument('--matcher_args',action='store', type=str, default='', dest='matcher_args') 
    parser.add_argument('--cross_check',action='store', type=bool, default=True, dest='cross_check') 

    #misc
    parser.add_argument('--print_every',action='store', type=int, default=1, dest='print_every') 

def PostprocessArgs(opts): 
    opts.ext = [x for x in opts.ext.split(',')]

if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    SetArguments(parser)
    opts = parser.parse_args()
    FeatMatch(opts)