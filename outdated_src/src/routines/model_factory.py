from src.models.deeplabv3_plus import DeepLab
from src.models.unet import UNet
from src.models.fcn import FCN

class ModelFactory:
    def __init__(self, model_chosen:str, num_classes:int) -> None:
        if model_chosen == "deeplab":
            deeplab = DeepLab("resnet101", "imagenet", num_classes, "softmax2d")
            self._model = deeplab.get_deeplab()

        elif model_chosen == "unet":
            unet = UNet("resnet101", "imagenet", num_classes, "softmax2d")
            self._model = unet.get_unet()
        
        elif model_chosen == "fcn":
            fcn = FCN()

        elif model_chosen == "mobile":
            mobile = None
        
        return None

    def get_model(self):
        return self._model
