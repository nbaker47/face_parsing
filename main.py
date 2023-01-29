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
    #path = input("Input path to root dataset directory (containing /train, /test folders)")
    ds_path = "/home/nathan/Documents/final_project/datasets/Helen/helenstar_release/"

    if mode == "train":

        model_chosen = input("Which segmenation model to train? \n DeepLabV3+ (deeplab) \nUNet (unet) \nFCN (fcn) \nMobileNet+ (mobile)")
        sample_size = 10    # take a sample of size x of dataset
        res = 128   # img res

        # Train
        X,Y, num_classes, rgb_vals = pp.read_dataset(ds_path, sample_size=sample_size, res=res, mode="train")
        t = Trainer(X,Y, batch_size=5)
        t.train(model_chosen=model_chosen, num_classes=num_classes, epochs=4)
    
    elif mode == "test":

        # model_path = input("Input path to root dataset directory (containing /train, /test folders)")
        model_path = '/home/nathan/Documents/final_project/saved_models/helen/helen_deeplab_no_aug.pth'
        sample_size = 10
        res = 256   # img res must be same as what it was trained for

        # Test
        X,Y, num_classes, rgb_vals = pp.read_dataset(ds_path, sample_size=sample_size, res=res, mode="test")
        t = Tester(X,Y)
        model = torch.load(model_path, map_location=DEVICE)
        print("model loaded")
        t.view_predictions(model, num_classes, rgb_vals, save=False)

    else:
        print("enter valid choice")

    print("success")

if __name__ == "__main__":
    sys.exit(main())