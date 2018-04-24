import cv2 
import numpy as np 
import pickle as pkl 
import argparse
#from itertools import izip
izip = zip
import pdb

from utils import * 

def get3dPtsRef(idx, mask, length): 
    out = (idx[mask], np.arange(np.sum(mask)))
    return out 

def getBaselineTriangulation(imgName1,imgName2,K):
    img1 = cv2.imread(imgName1)
    img2 = cv2.imread(imgName2)

    #Converting from BGR to RGB format
    img1 = img1[:,:,::-1]
    img2 = img2[:,:,::-1]

    #1. FEATURE MATCHING (ONLY BRUTE FORCE MATCHING IS IMPLEMENTED FOR NOW)..
    surfer=cv2.xfeatures2d.SURF_create()
    kp1, desc1 = surfer.detectAndCompute(img1,None)
    kp2, desc2 = surfer.detectAndCompute(img2,None)

    #Storing keypoitns and descriptors for future reference
    kps = [kp1, kp2]
    descs = [desc1, desc2]

    matcher = cv2.BFMatcher(crossCheck=True)
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key = lambda x:x.distance)

    #2. FUNDAMENTAL MATRIX ESTIMATION USING RANSAC + 8 POINT ALGORITHM
    img1pts,img2pts = GetAlignedMatches(kp1,desc1,kp2,desc2,matches)
    F,mask = cv2.findFundamentalMat(img1pts,img2pts,method=cv2.FM_RANSAC,param1=opts.outlierThres,
                                    param2=opts.fundProb)
    mask = mask.astype(bool).flatten()

    #Digression: Saving pointers from 2D features to 3D points' indices
    pts3dRef = list()
    pts3dRef.append(get3dPtsRef(np.array([x.queryIdx for x in matches]), mask, len(kp1)))
    pts3dRef.append(get3dPtsRef(np.array([x.trainIdx for x in matches]), mask, len(kp2)))

    #3. CAMERA POSE ESTIMATION
    E = K.T.dot(F.dot(K))
    _,R,t,_ = cv2.recoverPose(E,img1pts[mask],img2pts[mask],K)

    #4. TRIANGULATION. 
    img1ptsHom = cv2.convertPointsToHomogeneous(img1pts[mask])[:,0,:]
    img2ptsHom = cv2.convertPointsToHomogeneous(img2pts[mask])[:,0,:]

    img1ptsNorm = (np.linalg.inv(K).dot(img1ptsHom.T)).T
    img2ptsNorm = (np.linalg.inv(K).dot(img2ptsHom.T)).T

    img1ptsNorm = cv2.convertPointsFromHomogeneous(img1ptsNorm)[:,0,:]
    img2ptsNorm = cv2.convertPointsFromHomogeneous(img2ptsNorm)[:,0,:]

    pts4d = cv2.triangulatePoints(np.eye(3,4),np.hstack((R,t)),img1ptsNorm.T,img2ptsNorm.T)
    pts3d = cv2.convertPointsFromHomogeneous(pts4d.T)[:,0,:]

    return pts3d, pts3dRef, kps, descs

def get2D3DMatches(kps, descs, newkp, newdesc, pts3d, pts3dRef): 
    
    matcher = cv2.BFMatcher(crossCheck=True)
    out1, out2 = None, None
    
    for i, (_kp, _desc) in enumerate(izip(kps,descs)): 
        #picking only those descriptors whose 3D points are triangulated
        kp, desc = np.array(_kp)[pts3dRef[i][0]], np.array(_desc)[pts3dRef[i][0]]
        
        matches = matcher.match(desc, newdesc)
        
        #Filtering out matched 2D keypoitns from new view
        imgnidx = np.array([m.trainIdx for m in matches])
        kpn_ = (np.array(newkp))[imgnidx]
        imgnpts = np.array([kp.pt for kp in kpn_])
        
        #Filtering out already triangulated matched 3D points 
        imgiidx = np.array([m.queryIdx for m in matches])
        cloudpts = pts3d[pts3dRef[i][1][imgiidx]]
        
        if out1 is None: 
            out1 = imgnpts
            out2 = cloudpts
            
        else: 
            out1 = np.concatenate((out1,imgnpts),axis=0)
            out2 = np.concatenate((out2,cloudpts),axis=0)
            
    return imgnpts, cloudpts

def getNewCameraPose(imgName, pts3d, pts3dRef, kps, descs, newkp, newdesc, K):
    imgnpts, cloudpts = get2D3DMatches(kps, descs, newkp, newdesc, pts3d, pts3dRef)

    _, rnvec, tn, mask2 = cv2.solvePnPRansac(cloudpts[:,np.newaxis], imgnpts[:,np.newaxis], K, None,
                                          confidence=.99,flags=cv2.SOLVEPNP_DLS)

    Rn,_ = cv2.Rodrigues(rnvec)

    return Rn, tn, 
    

def addViewsToReconstruction(imgNames, pts3d, pts3dRef, kps, descs, K): 

    for imgName in imgNames: 

        imgn = cv2.imread(imgName)
        imgn = imgn[:,:,::-1]
        surfer=cv2.xfeatures2d.SURF_create()
        newkp, newdesc = surfer.detectAndCompute(imgn,None) 

        Rn, tn = getNewCameraPose(imgName, pts3d, pts3dRef, kps, descs, newkp, newdesc, K)


        break 

    return None

def main(opts, imgNames, fileName): 
    #Reading two images for reference
#    imgNames = ['../data/fountain-P11/images/0004.jpg','../data/fountain-P11/images/0005.jpg',
#                '../data/fountain-P11/images/0005.jpg']
    #imgNames = ['/root/FYP/SfM/data/fountain-P11/images/0004.jpg','/root/FYP/SfM/data/fountain-P11/images/0005.jpg',
    #            '/root/FYP/SfM/data/fountain-P11/images/0005.jpg']


    K = np.array([[2759.48,0,1520.69],[0,2764.16,1006.81],[0,0,1]]) #hardcoded for now, have to generalize.. 

    #Initial 2 view SFM
    _pts3d, _pts3dRef, kps, descs = getBaselineTriangulation(imgNames[0], imgNames[1],K)

    #Incrementally add more cameras now
    addViewsToReconstruction(imgNames[2:], _pts3d, _pts3dRef, kps, descs, K)

#    return 

    #Finally, saving 3d points in .ply format to view in meshlab software
    pts2ply(_pts3d, filename=fileName)
    return 

def SetArguments(parser): 
    parser.add_argument('runserver',action='store',type=str,default='') 
    parser.add_argument('0.0.0.0:8000',action='store',type=str,default='') 
    parser.add_argument('-dataDir',action='store',type=str,default='../data/fountain-P11/images/keypoints_descriptors',dest='dataDir') 
    parser.add_argument('-outName',action='store',type=str,default='../data/fountain-P11/images/matches',dest='outDir') 
    parser.add_argument('-printEvery',action='store', type=int, default=1, dest='printEvery') 
    parser.add_argument('-crossCheck',action='store', type=bool, default=True, dest='crossCheck') 

    parser.add_argument('-outlierThres',action='store', type=float, default=.1, dest='outlierThres') 
    parser.add_argument('-fundProb',action='store', type=float, default=.99, dest='fundProb') 
    return 

#if __name__=='__main__': 
parser = argparse.ArgumentParser()
SetArguments(parser)
opts = parser.parse_args()
#main(opts)
