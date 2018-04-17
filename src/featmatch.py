import cv2 
import numpy as np 
import pickle 
import argparse
import os 

from utils import * 

def FeatMatch(opts): 
    img_names = sorted(os.listdir(opts.data_dir))
    img_paths = [os.path.join(opts.data_dir, x) for x in img_names[:3]]
    
    feat_out_dir = os.path.join(opts.out_dir,'features')
    if not os.path.exists(feat_out_dir): 
        os.makedirs(feat_out_dir)

    matches_out_dir = os.path.join(opts.out_dir,'matches')
    if not os.path.exists(matches_out_dir): 
        os.makedirs(matches_out_dir)

    data = []

    for i, img_path in enumerate(img_paths): 
        img = cv2.imread(img_path)
        img_name = img_names[i].split('.')[0]
        img = img[:,:,::-1]

        feat = getattr(cv2.xfeatures2d, '{}_create'.format(opts.features))()
        kp, desc = feat.detectAndCompute(img,None)
        data.append((img_name, kp, desc))

        kp_ = SerializeKeypoints(kp)
        
        with open(os.path.join(feat_out_dir, 'kp_{}.pkl'.format(img_name)),'wb') as out:
            pickle.dump(kp_, out)

        with open(os.path.join(feat_out_dir, 'desc_{}.pkl'.format(img_name)),'wb') as out:
            pickle.dump(desc, out)

    for i in xrange(len(data)): 
        for j in xrange(i+1, len(data)): 
            img_name1, kp1, desc1 = data[i]
            img_name2, kp2, desc2 = data[j]

            matcher = getattr(cv2,opts.matcher)(crossCheck=opts.cross_check)
            matches = matcher.match(desc1,desc2)

            matches = sorted(matches, key = lambda x:x.distance)
            matches_ = SerializeMatches(matches)

            pickle_path = os.path.join(matches_out_dir, 'match_{}_{}.pkl'.format(img_name1, img_name2))
            with open(pickle_path,'wb') as out:
                pickle.dump(matches_, out)

def SetArguments(parser): 

    #directories stuff
    parser.add_argument('--data_dir',action='store',type=str,default='../data/fountain-P11/images/',
                        dest='data_dir') 
    parser.add_argument('--ext',action='store',type=str,default='jpg,png',dest='ext') 
    parser.add_argument('--out_dir',action='store',type=str,default='../data/fountain-P11/',
                        dest='out_dir') 

    #feature matching args
    parser.add_argument('--features',action='store', type=str, default='SURF', dest='features') 
    parser.add_argument('--matcher',action='store', type=str, default='BFMatcher', dest='matcher') 
    parser.add_argument('--cross_check',action='store', type=bool, default=True, dest='cross_check') 
    parser.add_argument('--matcher_args',action='store', type=str, default='', dest='matcher_args') 
    
    #misc
    parser.add_argument('--print_every',action='store', type=int, default=1, dest='print_every') 

def PostprocessArgs(opts): 
    opts.ext = [x for x in opts.ext.split(',')]

if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    SetArguments(parser)
    opts = parser.parse_args()
    FeatMatch(opts)