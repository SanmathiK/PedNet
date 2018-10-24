import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
import matplotlib.image as mpimg
import os
from imageio import imread
import matplotlib.pyplot as plt

def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

class RawTrainingData():
    '''Class to load raw training data'''

    def __init__(self, train_img_path, annotation_path, classes):
        self.train_img_path = train_img_path
        self.annotation_path = annotation_path
        self.classes = classes
        self.img_files = sorted(os.listdir(self.train_img_path))
        self.ann_files = sorted(os.listdir(self.annotation_path))
        self.N = len(self.img_files)
        sample_image = imread(os.path.join(self.train_img_path, self.img_files[0]))
        self.H, self.W, self.C = sample_image.shape
        self.X = np.zeros((self.N, self.H, self.W))
        self.X_mask = np.zeros((self.N, self.H, self.W))
        self.y = np.zeros((self.N, 1))

    def load_data(self):
        '''Loading Image Data into numpy arrays'''
        for i in range(self.N):
            if i % 50 == 0:
                print("Loaded %d images out of %d..." %(i+50, self.N))
            img_path = os.path.join(self.train_img_path, self.img_files[i])
            ann_path = os.path.join(self.annotation_path, self.ann_files[i])
            img = imread(img_path)
            img = rgb2ycbcr(img)
            img = img[:, :, 0]
            if img.shape[0] == 640 and img.shape[1] == 480:
                self.X[i] = img.T
            else:
                self.X[i] = img

            ann = imread(ann_path)
            if ann.shape[0] == 640 and ann.shape[1] == 480:
                self.X_mask[i] = ann.T
            else:
                self.X_mask[i] = ann
            self.y[i] = int(i/300)

class PreProcTransforms:

    def horizontal_shift(self, img, ann, x_shift, y_shift):
        shifted_image = np.roll(img, x_shift, axis=1)
        shifted_ann = np.roll(ann, x_shift, axis=1)
        shifted_image = np.roll(shifted_image, y_shift, axis=0)
        shifted_ann = np.roll(ann, y_shift, axis=0)
        return shifted_image, shifted_ann


    def gaussian_blurr(self, img, sigma):
        blur_image = ndimage.gaussian_filter(img, sigma=sigma, mode='reflect')
        return img, ann

    def change_bg(self, img, ann, delta):
        bkg = img.copy()
        bkg = bkg + delta
        bkg[ann==0] = 0
        frg = img.copy()
        frg[ann!=0] = 0
        img1 = bkg+frg
        return img1, ann
