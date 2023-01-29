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

    # helper function for data visualization
    def visualize(self, original_images, ground_truth_masks, predicted_masks, ious):
        
        figure, axis = plt.subplots(5, 3, figsize=(10,10))
        figure.suptitle('Original image                       Ground Truth                      Predicted', 
             fontsize=14, fontweight='bold')

        for x in range(len(original_images)):
            axis[x,0].imshow(original_images[x])
            axis[x,0].set_xticks([])
            axis[x,0].set_yticks([])
            axis[x,1].imshow(ground_truth_masks[x])
            axis[x,1].set_xticks([])
            axis[x,1].set_yticks([])
            axis[x,2].title.set_text("IoU: " + str(ious[x]))
            axis[x,2].imshow(predicted_masks[x])
            axis[x,2].set_xticks([])
            axis[x,2].set_yticks([])

        return figure

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
        original_images = []
        ground_truth_masks = []
        predicted_masks = []

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

            # display images
            if idx < 5:
                original_images.append(image_vis[:,:,0])
                ground_truth_masks.append(gt_mask[0,:,:,])
                predicted_masks.append(pred_mask[0,:,:,])

            gts.append(gt_mask)
            preds.append(pred_mask)

            if save == True:
                # Save Predictions for use in Label Adapter
                im = Image.fromarray(pred_mask.squeeze(0).astype(np.uint8))
                im.save("/content/drive/MyDrive/helen_deeplab_no_aug_out/" + str(idx) + "_img.png")

                im2 = Image.fromarray(gt_mask.squeeze(0).astype(np.uint8))
                im2.save("/content/drive/MyDrive/helen_deeplab_no_aug_out/" + str(idx) + "_seg.png")
        

        figure = self.visualize(original_images, ground_truth_masks, predicted_masks, ious)
        figure.tight_layout()
        plt.show()

        fs1_numpy = np.array(f1s)
        av_f1s = np.average(fs1_numpy, axis=0, weights=(fs1_numpy > 0))
        av_f1s_av = av_f1s.mean(axis=0)
        iou_av = self.average(ious)

        print ("Dataset F1 = ", av_f1s)
        print ("Dataset F1 av = ", av_f1s_av)
        print ("Dataset IoU av = ", iou_av)


