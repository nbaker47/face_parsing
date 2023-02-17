import segmentation_models_pytorch as smp


class DeepLab:
    def __init__(self, encoder, encoder_weights, num_classes, activation):
        
        #DEEPLABV3+
        self._deeplab = smp.DeepLabV3Plus(
            in_channels=1,
            encoder_name=encoder, 
            encoder_weights=encoder_weights, 
            classes=num_classes, 
            activation=activation,
        )
        
        # Preprocess encoder weights
        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

        return None

    def get_deeplab(self):
        return self._deeplab