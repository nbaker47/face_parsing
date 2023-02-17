from src.routines.preprocessor import Preprocessor
from src.routines.trainer import Trainer
from src.routines.tester import Tester
from torch.utils.data import DataLoader
import sys
import torch

def main():
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read DS from disk and preprocess
    pp = Preprocessor()

    mode = input("Train or test model? ('train'/'test') \n")
    #ds_path_img = input("Input path to root to images")
    #ds_path_seg = input("Input path to root to masks")
    ds_path_img = "/home/nathan/Documents/final_project/datasets/LaPa/train/images/"
    ds_path_seg = "/home/nathan/Documents/final_project/datasets/LaPa/train/labels/"

    if mode == "train":

        model_chosen = input("Which segmenation model to train? \nDeepLabV3+ (deeplab) \nUNet (unet) \nFCN (fcn) \nMobileNet+ (mobile)\n")
        sample_size = 18168    # take a sample of size x of dataset
        res = 128   # img res

        # Train
        X,Y, num_classes, rgb_vals = pp.read_dataset(ds_path_img, ds_path_seg, sample_size=sample_size, res=res, mode="train")
        t = Trainer(X,Y, batch_size=2)
        t.train(model_chosen=model_chosen, num_classes=num_classes, epochs=100)
    
    elif mode == "test":

        #model_path = input("Input path to model to test")
        model_path = '/home/nathan/Documents/final_project/saved_models/helen/helen_deeplab_no_aug.pth'
        sample_size = 10
        res = 128   # img res must be same as what it was trained for

        # Test
        X,Y, num_classes, rgb_vals = pp.read_dataset(ds_path_img, ds_path_seg, sample_size=sample_size, res=res, mode="test")
        t = Tester(X,Y)
        model = torch.load(model_path, map_location=DEVICE)
        print("model loaded")
        t.view_predictions(model, num_classes, rgb_vals, save=False)

    else:
        print("enter valid choice")

    print("success")

if __name__ == "__main__":
    sys.exit(main())