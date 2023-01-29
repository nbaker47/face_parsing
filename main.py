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
    #path = "/home/nathan/Documents/final_project/datasets/Helen/helenstar_release/"

    if mode == "train":
        # Train
        X,Y, num_classes, rgb_vals = pp.read_dataset("/home/nathan/Documents/final_project/datasets/Helen/helenstar_release/", sample_size=10, res=128, mode="train")
        t = Trainer(X,Y, batch_size=5)
        t.train(model_chosen="deeplab", num_classes=num_classes, epochs=4)
    
    elif mode == "test":
        # Test
        X,Y, num_classes, rgb_vals = pp.read_dataset("/home/nathan/Documents/final_project/datasets/Helen/helenstar_release/", sample_size=10, res=256, mode="test")
        t = Tester(X,Y)
        model = torch.load('/home/nathan/Documents/final_project/saved_models/helen/helen_deeplab_aug5 (1).pth', map_location=DEVICE)
        print("model loaded")
        t.view_predictions(model, num_classes, rgb_vals, save=False)

    else:
        print("enter valid choice")

    print("success")

if __name__ == "__main__":
    sys.exit(main())