import cv2
import glob
import torch
import numpy as np
import os
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random

class Preprocessor:

    def __init__(self):
        pass

    def __one_hot_permute(self, X,Y, num_classes):
        one_hot_Y = torch.nn.functional.one_hot(Y.to(torch.int64), num_classes)
        X = X.unsqueeze(1).float()
        Y = one_hot_Y.permute(0, 3, 1, 2).float()
        return X,Y

    """Read dataset from disk"""
    def read_dataset(self, dataset_root_img:str, dataset_root_seg:str, sample_size:int=1000, res:int=128, normalize=False, mode="train"):

        """IMG"""
        #Capture training image info as a list
        train_images = []
        for count, img_path in enumerate (sorted(glob.glob(os.path.join(dataset_root_img, "*.jpg")))):

            #Process images
            img = cv2.imread(img_path)
            img = cv2.resize(img, (res, res))

            """Normalise"""
            # convert to gray
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            if normalize == True:
                # blur
                smooth = cv2.GaussianBlur(gray, (95,95), 0)
                # divide gray by morphology image
                division = cv2.divide(gray, smooth, scale=192)
                train_images.append(division)
            else:
                train_images.append(gray)

            # Stop after SAMPLE_SIZE is reached
            print("X:", str(count), "/", str(sample_size), end='\r')
            if count > sample_size:
                break

        """MASK"""
        #Capture mask/label info as a list
        train_masks = [] 
        for count, mask_path in enumerate(sorted(glob.glob(os.path.join(dataset_root_seg, "*.png")))):

            #Process images
            mask = cv2.imread(mask_path, 0)    
            mask = cv2.resize(mask, (res, res))
            train_masks.append(mask)   

            # Stop after SAMPLE_SIZE is reached
            print("Y:", str(count), "/", str(sample_size), end='\r')
            if count > sample_size:
                break

        if mode == "test":
            # shuffle testing images, training images will be shuffled when train/test splitted
            c = list(zip(train_images, train_masks))
            random.shuffle(c)
            train_images, train_masks = zip(*c)

        # Conver to Torch Tensor
        X = torch.tensor(np.array(train_images))
        Y = torch.tensor(np.array(train_masks))
        # Find unique RGB class codes
        rgb_vals = np.unique(ar = Y.numpy())
        num_classes=len(rgb_vals)
        # One hot Y
        X,Y = self.__one_hot_permute(X,Y, num_classes) 
        
        return X,Y,num_classes, rgb_vals

    """Augment images"""
    def augment(self, x, y, res):
        # rotate XY
        r = random.randint(-18,18)
        x = F.rotate(x, r)
        y = F.rotate(y, r)

        # persepective warp XY with percentage: 40% WARP
        prob_warp = random.randint(0,10)
        starting_coords = [[0,res],[res,res], [0,0], [res,0]]
        end_coords = starting_coords
        if prob_warp > 7:
        
            for coord in end_coords:
                for val in coord:
                    transformation = random.random()
                    val = val * transformation

            x, y = F.perspective(x, starting_coords, end_coords), F.perspective(y, starting_coords, end_coords)

        # (Gaussian) blurs XY with percentage: 30% BLUR
        prob_blur = random.randint(0,10)
        if prob_blur > 0.7:
            blurrer = T.GaussianBlur(kernel_size=(3,5))
            x, y = blurrer(x), blurrer(y)

        #modulations to brightness and contrast X with percentage: 30% MODULATION
        prob_mod = random.randint(0,10)
        if prob_mod > 0.7:
            jitter = T.ColorJitter()
            x = jitter(x)
        
        #modulations to brightness and contrast X with percentage: 30% MODULATION
        #prob_mod = random.randint(0,10)
        #if prob_mod > 0.7:
        #  x = F.hflip(x)
        #  y = F.hflip(y)

        #conversion to grayscale X
        x = T.Grayscale()(x)
        
        return x,y



