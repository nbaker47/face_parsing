import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split
from src.routines.model_factory import ModelFactory
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from segmentation_models_pytorch import utils as smp_utils


class TrainDS(torch.utils.data.Dataset):
  def __init__(self, x, y) -> None:
    super(TrainDS, self).__init__()
    
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

class Trainer:

  # Set global device to GPU
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  """"CONSTRUCTOR"""
  def __init__(self, X, Y, batch_size=5) -> None:
    # Split our DS
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

    # DS
    self._train_ds = TrainDS(x_train, y_train)
    self._val_ds = TrainDS(x_test, y_test)

    # Get train and val data loaders
    self._train_loader = DataLoader(self._train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    self._valid_loader = DataLoader(self._val_ds, batch_size=batch_size, shuffle=False, num_workers=1)
  
    # Training few hyper parameters
    self.loss = loss = smp_utils.losses.BCELoss()
    self.metrics = metrics = [smp_utils.metrics.IoU(threshold=0.5)]

    
  """ TRAINING GRAPH PLOTTER """
  def show_graph(self, epoch_count, t_bce, v_bce, t_ious, v_ious):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 5))
    axes[0].plot(epoch_count,t_bce, label="Train ce")
    axes[0].plot(epoch_count,v_bce, label="Valid ce")
    axes[0].legend(loc="upper left")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("BCE Loss")

    axes[1].plot(epoch_count,t_ious, label="Train IoU")
    axes[1].plot(epoch_count,v_ious, label="Valid IoU")
    axes[1].legend(loc="upper left") 
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("IoU")
    
    fig.tight_layout()
    plt.show()


  """MODEL TRAINER"""
  def train(self, model_chosen="deeplab", num_classes=11, epochs=15):

    mf = ModelFactory(model_chosen, num_classes)
    model = mf.get_model()

    #clear CUDA cache
    torch.cuda.empty_cache()

    # define optimizer
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.001),
    ])

    # define learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    )

    #define train ecpochs w/ our hyperparams
    train_epoch = smp_utils.train.TrainEpoch(
        model, 
        loss=self.loss, 
        metrics=self.metrics, 
        optimizer=optimizer,
        device=self.DEVICE,
        verbose=True,
    )

    #define train ecpochs w/ our hyperparams
    valid_epoch = smp_utils.train.ValidEpoch(
        model, 
        loss=self.loss, 
        metrics=self.metrics, 
        device=self.DEVICE,
        verbose=True,
    )

    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    #Keep track of epoch BCE and IoU for graph plots
    epoch_count = []
    v_bce = []
    v_ious = []

    t_bce = []
    t_ious = []

    for i in range(epochs):

        # Perform training & validation
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(self._train_loader)
        valid_logs = valid_epoch.run(self._valid_loader)
        #log
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(model, str(model_chosen))
            print('Model saved!')

        # Append results to graphs:
        v_bce.append(valid_logs['bce_loss'])
        v_ious.append(valid_logs['iou_score'])
        t_bce.append(train_logs['bce_loss'])
        t_ious.append(train_logs['iou_score'])
        epoch_count.append(i)
    
    self.show_graph(epoch_count, t_bce, v_bce, t_ious, v_ious)


