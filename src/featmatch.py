import cv2 
import numpy as np 
import pickle 
import argparse
import os 
import multiprocessing

from utils import * 

def FeatMatchThreaded(opts, img_path, i, img_names):
    img = cv2.imread(img_path)
    img_name = img_names[i].split('.')[0]
    img = img[:,:,::-1]

    feat = getattr(cv2.xfeatures2d, '{}_create'.format(opts.features))()
    kp, desc = feat.detectAndCompute(img,None)

    kp_ = SerializeKeypoints(kp)
    
    with open(os.path.join(feat_out_dir, 'kp_{}.pkl'.format(img_name)),'wb') as out:
        pickle.dump(kp_, out)

    with open(os.path.join(feat_out_dir, 'desc_{}.pkl'.format(img_name)),'wb') as out:
        pickle.dump(desc, out)

    if opts.save_results: 
        raise NotImplementedError

    if (i % opts.print_every) == 0:
        print('{}/{} features done..'.format(i+1,len(img_paths)))

def FeatMatch(opts, data_files=[]): 
    
    if len(data_files) == 0: 
        img_names = sorted(os.listdir(opts.data_dir))
        img_paths = [os.path.join(opts.data_dir, x) for x in img_names if \
                    x.split('.')[-1] in opts.ext]
    
    else: 
        img_paths = data_files
        img_names = sorted([x.split('/')[-1] for x in data_files])
        
    feat_out_dir = os.path.join(opts.out_dir,'features',opts.features)
    if not os.path.exists(feat_out_dir): 
        os.makedirs(feat_out_dir)
    
    processes = list()
    for i, img_path in enumerate(img_paths): 
        process = multiprocessing.Process(target=FeatMatchThreaded, args=(opts, img_path, i, img_names))
        processes.append(process)

    for process in processes:
        print("Process starting...")
        process.start()
    
    results = [output.get() for p in processes]

    for process in processes:
        print("Waiting for process...")
        process.join()

def SetArguments(parser): 

    #directories stuff
    parser.add_argument('--data_files',action='store',type=str,default='',dest='data_files') 
    parser.add_argument('--data_dir',action='store',type=str,default='../data/fountain-P11/images/',
                        dest='data_dir') 
    parser.add_argument('--ext',action='store',type=str,default='jpg,png',dest='ext') 
    parser.add_argument('--out_dir',action='store',type=str,default='../data/fountain-P11/',
                        dest='out_dir') 

    #feature matching args
    parser.add_argument('--features',action='store', type=str, default='SURF', dest='features') 
    
    #misc
    parser.add_argument('--print_every',action='store', type=int, default=1, dest='print_every')
    parser.add_argument('--save_results',action='store', type=str, default=False, dest='save_results')  

def PostprocessArgs(opts): 
    opts.ext = [x for x in opts.ext.split(',')]
    
    opts.data_files_ = []
    if opts.data_files != '': 
        opts.data_files_ = opts.data_files.split(',')
    opts.data_files = opts.data_files_

if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    SetArguments(parser)
    opts = parser.parse_args()
    PostprocessArgs(opts)
    FeatMatch(opts, opts.data_files)
