import numpy as np
import cv2
import torch
from torch import nn
from torch.utils import model_zoo

from .base import Base


class ECCV16Model(Base):

    def __init__(self, norm_layer=nn.BatchNorm2d, pretrained=None, caffe=False):
        super(ECCV16Model, self).__init__()

        self.pretrained = pretrained
        if (pretrained is not None):
            if (not isinstance(pretrained, dict)):
                raise TypeError('`pretrained` must be dict or None.')
        else:
            msg = 'training not yet supported, `pretrained` must be dict '
            msg += 'containing pretrained model files'
            raise NotImplementedError(msg)

        self.caffe = caffe
        # Load caffe model
        if (self.caffe):
            # Check if necessary keys exist in dict
            self._validate_keys(
                pretrained,
                ['prototxt', 'model', 'points'],
                argname='pretrained')
            # Load caffe model
            self.model = self.load_caffe_model(
                pretrained['prototxt'],
                pretrained['model'],
                pretrained['points'])

        # Initialize Pytorch layers
        else:
            self.model1 = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(True),
                norm_layer(64)
            )

            self.model2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(True),
                norm_layer(128)
            )

            self.model3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(True),
                norm_layer(256)
            )

            self.model4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                norm_layer(512)
            )

            self.model5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
                nn.ReLU(True),
                norm_layer(512)
            )

            self.model6 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
                nn.ReLU(True),
                norm_layer(512)
            )

            self.model7 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                norm_layer(512)
            )

            self.model8 = nn.Sequential(*[
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True)
            ])

            self.softmax = nn.Softmax(dim=1)
            self.model_out = nn.Conv2d(
                313, 2,
                kernel_size=1, padding=0,
                dilation=1, stride=1, bias=False)
            self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

            # Load pretrained state
            self._validate_keys(pretrained, 'model', argname='pretrained')
            if (pretrained['model'].startswith('http')):
                self.pretrained_state = model_zoo.load_url(
                    pretrained['model'], map_location='cpu', check_hash=True)
            else:
                self.pretrained_state = torch.load(pretrained['model'])

            self.load_state_dict(self.pretrained_state)
            self.eval()


    def load_caffe_model(self, prototxt_path, model_path, points_path):
        """Load serialized Caffe colorization model"""
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        pts = np.load(points_path)
        # Add the cluster centers as 1x1 convolutions to the model
        class8 = net.getLayerId('class8_ab')
        conv8 = net.getLayerId('conv8_313_rh')
        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype('float32')]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype='float32')]

        return net


    def forward(self, image_l):

        # Forward pass through caffe model
        if (self.caffe):
            # Convert torch tensor to Numpy array
            image_l = image_l.detach().numpy().squeeze()
            # Center channel by removing mean
            image_l = image_l - np.mean(image_l)
            # Model takes L channel as input
            self.model.setInput(cv2.dnn.blobFromImage(image_l.squeeze()))
            # Model predicts a and b channels
            image_ab = self.model.forward()
            return torch.Tensor(image_ab)

        # Forward pass through PyTorch model
        else:
            conv1_2 = self.model1(self.normalize_l(image_l))
            conv2_2 = self.model2(conv1_2)
            conv3_3 = self.model3(conv2_2)
            conv4_3 = self.model4(conv3_3)
            conv5_3 = self.model5(conv4_3)
            conv6_3 = self.model6(conv5_3)
            conv7_3 = self.model7(conv6_3)
            conv8_3 = self.model8(conv7_3)
            out_reg = self.model_out(self.softmax(conv8_3))

            return self.denormalize_ab(self.upsample4(out_reg))


if __name__ == '__main__':
    # Checkpoint available at
    # https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth
    checkpoint = dict(model='weights/eccv16-9b330a0b.pth')
    model = ECCV16Model(pretrained=checkpoint)