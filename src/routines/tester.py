import torch
import tensorflow as tf
from torchmetrics.classification import F1Score, BinaryF1Score, MulticlassF1Score, MultilabelF1Score
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class TestDS(torch.utils.data.Dataset):
  def __init__(self, x, y) -> None:
    super(TestDS, self).__init__()
    
    # store the augmented tensors tensors
    self._x, self._y = x,y

  def __len__(self):
    # a DataSet must know it size
    return self._x.shape[0]

  def __getitem__(self, index):
    x = self._x[index, :]
    y = self._y[index, :]
    # print("GETTING ITEM")
    return x, y


class Tester:

    # Set global device to GPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self,X,Y) -> None:
        self.test_dataset = TestDS(X,Y)
    
    def average(self, lst):
        return sum(lst) / len(lst)

    # Helper functions from "https://github.com/HardevKhandhar/building-segmentation-image-processing/blob/main/geospatial_buildings.ipynb"
    # helper function for data visualization
    def visualize(self, **images):
        """
        Plot images in one row
        """
        n_images = len(images)
        plt.figure(figsize=(20,8))
        for idx, (name, image) in enumerate(images.items()):
            plt.subplot(1, n_images, idx + 1)
            plt.xticks([]); 
            plt.yticks([])
            # get title from the parameter names
            plt.title(name.replace('_',' ').title(), fontsize=20)
            plt.imshow(image)
        plt.show()

    def reverse_one_hot(self, image):
        """
        Transform a 2D array in one-hot format (depth is num_classes),
        to a 2D array with only 1 channel, where each pixel value is
        the classified class key.
        # Arguments
            image: The one-hot format image 
            
        # Returns
            A 2D array with the same width and hieght as the input, but
            with a depth size of 1, where each pixel value is the classified 
            class key.
        """
        #argmax returns the indices of the maximum values along an axis. 
        x = np.argmax(image, axis = -3)
        return x

    # Perform colour coding on the reverse-one-hot outputs
    def colour_code_segmentation(self, image, label_values):
        """
        Given a 1-channel array of class keys, colour code the segmentation results.
        # Arguments
            image: single channel array where each value represents the class key.
            label_values

        # Returns
            Colour coded image for segmentation visualization
        """
        colour_codes = np.array(label_values)
        #print(colour_codes)
        x = colour_codes[image.numpy().astype(int)]

        return x

    def view_predictions(self, model, numm_classes, rgb_vals, save=False):

        ious = []
        f1s = []
        gts = []
        preds = []

        ds = self.test_dataset

        #predict
        for idx in range(len(ds)):

            # retreive images
            image, gt_mask = ds[idx]
            image_vis = ds[idx][0]
            image_vis = np.transpose(image_vis,(1,2,0))
            
            x_tensor = image.to(self.DEVICE).unsqueeze(0)
            # Predict test image
            pred_mask = model(x_tensor).detach().cpu().numpy()
            gt_mask=gt_mask.unsqueeze(0)

            # get f1 score for each class
            metric = MultilabelF1Score(num_labels=numm_classes, average=None, labels=np.unique(pred_mask))
            f1 = metric(torch.tensor(pred_mask), torch.tensor(gt_mask))
            if len(f1) == numm_classes:
                f1s.append(np.array(f1))

            # Get prediction channel corresponding to face
            pred_mask = self.colour_code_segmentation(self.reverse_one_hot(torch.tensor(pred_mask)), rgb_vals)
            
            # Convert gt_mask from `CHW` format to `HWC` format
            gt_mask = self.colour_code_segmentation(self.reverse_one_hot(torch.tensor(gt_mask)), rgb_vals)
            
            #get overall IoU score
            m = tf.keras.metrics.MeanIoU(num_classes=numm_classes)
            m.update_state(gt_mask, pred_mask)
            iou = m.result().numpy()
            ious.append(iou)

            # display first 5 images
            if [idx][0] < 1:
                print(f1)
                self.visualize(
                    original_image = image_vis[:,:,0],
                    ground_truth_mask = gt_mask[0,:,:,],
                    predicted_mask = pred_mask[0,:,:,],
                )

            gts.append(gt_mask)
            preds.append(pred_mask)

            if save == True:
                # Save Predictions for use in Label Adapter
                im = Image.fromarray(pred_mask.squeeze(0).astype(np.uint8))
                im.save("/content/drive/MyDrive/helen_deeplab_no_aug_out/" + str(idx) + "_img.png")

                im2 = Image.fromarray(gt_mask.squeeze(0).astype(np.uint8))
                im2.save("/content/drive/MyDrive/helen_deeplab_no_aug_out/" + str(idx) + "_seg.png")
        
        fs1_numpy = np.array(f1s)
        av_f1s = np.average(fs1_numpy, axis=0, weights=(fs1_numpy > 0))
        av_f1s_av = av_f1s.mean(axis=0)
        iou_av = self.average(ious)

        print ("Dataset F1 = ", av_f1s)
        print ("Dataset F1 av = ", av_f1s_av)
        print ("Dataset IoU av = ", iou_av)


