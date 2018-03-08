import cv2
import numpy as np
import pickle as pkl
import argparse

import pdb

from utils import *

# Read all images from the given directory.


def readAllImages(path):
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)
    return images


def main(opts):
    # Reading two images for reference
    img1 = cv2.imread('../data/fountain-P11/images/0004.jpg')
    img2 = cv2.imread('../data/fountain-P11/images/0006.jpg')

    # Converting from BGR to RGB format
    img1 = img1[:, :, ::-1]
    img2 = img2[:, :, ::-1]

    # 1. FEATURE MATCHING (ONLY BRUTE FORCE MATCHING IS IMPLEMENTED FOR NOW)..
    surfer = cv2.xfeatures2d.SURF_create()
    kp1, desc1 = surfer.detectAndCompute(img1, None)
    kp2, desc2 = surfer.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher(crossCheck=True)
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 2. FUNDAMENTAL MATRIX ESTIMATION USING RANSAC + 8 POINT ALGORITHM
    img1pts, img2pts = GetAlignedMatches(kp1, desc1, kp2, desc2, matches)
    F, mask = cv2.findFundamentalMat(img1pts, img2pts, method=cv2.FM_RANSAC, param1=opts.outlierThres,
                                     param2=opts.fundProb)
    mask = mask.astype(bool).flatten()

    # 3. CAMERA POSE ESTIMATION
    # hardcoded for now, have to generalize..
    K = np.array([[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0, 1]])
    E = K.T.dot(F.dot(K))
    _, R, t, _ = cv2.recoverPose(E, img1pts[mask], img2pts[mask], K)

    # 4. TRIANGULATION.
    img1ptsHom = cv2.convertPointsToHomogeneous(img1pts[mask])[:, 0, :]
    img2ptsHom = cv2.convertPointsToHomogeneous(img2pts[mask])[:, 0, :]

    img1ptsNorm = (np.linalg.inv(K).dot(img1ptsHom.T)).T
    img2ptsNorm = (np.linalg.inv(K).dot(img2ptsHom.T)).T

    img1ptsNorm = cv2.convertPointsFromHomogeneous(img1ptsNorm)[:, 0, :]
    img2ptsNorm = cv2.convertPointsFromHomogeneous(img2ptsNorm)[:, 0, :]

    pts4d = cv2.triangulatePoints(np.eye(3, 4), np.hstack(
        (R, t)), img1ptsNorm.T, img2ptsNorm.T)
    pts3d = cv2.convertPointsFromHomogeneous(pts4d.T)[:, 0, :]

    # 6. Read the third images and apply SURF.
    img3 = cv2.imread('../data/fountain-P11/images/0005.jpg')
    img3 = img3[:, :, ::-1]
    surfer = cv2.xfeatures2d.SURF_create()
    kp3, desc3 = surfer.detectAndCompute(img3, None)

    img3pts, pts3dpts = ut.Find2D3DMatches(
        desc1, img1idx, desc2, img2idx, desc3, kp3, mask, pts3d)

    # 7. Apply the PnP algoritm.

    # 7.1 Apply RANSAC
    retval, Rvec, tnewgt, mask3gt = cv2.solvePnPRansac(pts3dpts[:, np.newaxis], img3pts[:, np.newaxis],
                                                       K, None, confidence=.99, flags=cv2.SOLVEPNP_DLS)
    Rnewgt, _ = cv2.Rodrigues(Rvec)

    Rnew, tnew, mask3 = sfmnp.LinearPnPRansac(
        pts3dpts, img3pts, K, outlierThres=5.0, iters=2000)

    # 8. Perfrom re-triangulation with the third image.
    kpNew, descNew = kp3, desc3

    kpOld, descOld = kp1, desc1
    ROld, tOld = np.eye(3), np.zeros((3, 1))

    accPts = []
    for (ROld, tOld, kpOld, descOld) in [(np.eye(3), np.zeros((3, 1)), kp1, desc1), (R, t, kp2, desc2)]:

        # Matching between old view and newly registered view..
        print '[Info]: Feature Matching..'
        matcher = cv2.BFMatcher(crossCheck=True)
        matches = matcher.match(descOld, desc3)
        matches = sorted(matches, key=lambda x: x.distance)
        imgOldPts, imgNewPts, _, _ = ut.GetAlignedMatches(kpOld, descOld, kpNew,
                                                          descNew, matches)

        # Pruning the matches using fundamental matrix..
        print '[Info]: Pruning the Matches..'
        F, mask = cv2.findFundamentalMat(
            imgOldPts, imgNewPts, method=cv2.FM_RANSAC, param1=.1, param2=.99)
        mask = mask.flatten().astype(bool)
        imgOldPts = imgOldPts[mask]
        imgNewPts = imgNewPts[mask]

        # Triangulating new points
        print '[Info]: Triangulating..'
        newPts = sfmnp.GetTriangulatedPts(
            imgOldPts, imgNewPts, K, Rnew, tnew, cv2.triangulatePoints, ROld, tOld)

        # Adding newly triangulated points to the collection
        accPts.append(newPts)

    # Append to the previous results.
    accPts.append(pts3d)

    # Finally, saving 3d points in .ply format to view in meshlab software
    ut.pts2ply(np.concatenate((accPts), axis=0), 'test.ply')

    return


def SetArguments(parser):
    parser.add_argument('-dataDir', action='store', type=str,
                        default='../data/fountain-P11/images/keypoints_descriptors', dest='dataDir')
    parser.add_argument('-outName', action='store', type=str,
                        default='../data/fountain-P11/images/matches', dest='outDir')
    parser.add_argument('-printEvery', action='store',
                        type=int, default=1, dest='printEvery')
    parser.add_argument('-crossCheck', action='store',
                        type=bool, default=True, dest='crossCheck')

    parser.add_argument('-outlierThres', action='store',
                        type=float, default=.1, dest='outlierThres')
    parser.add_argument('-fundProb', action='store',
                        type=float, default=.99, dest='fundProb')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    SetArguments(parser)
    opts = parser.parse_args()
    main(opts)
