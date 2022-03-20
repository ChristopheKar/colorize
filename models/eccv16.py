import warnings

import numpy as np
import cv2
import torch
from torch import nn
from torch.utils import model_zoo

from .base import Base


class ECCV16Model(Base):

    def __init__(self, norm_layer=nn.BatchNorm2d, checkpoint=None, caffe=False):
        super().__init__()

        # Set default model paths
        self.caffe = caffe
        if (self.caffe):
            # Set caffe model file paths
            weights_dir = 'weights/zhang_eccv16/caffe'
            prototxt_path = os.path.join(
                weights_dir, 'colorization_deploy_v2.prototxt')
            points_path = os.path.join(
                weights_dir, 'pts_in_hull.npy')
            checkpoint_path = os.path.join(
                weights_dir, 'colorization_release_v2.caffemodel')
        else:
            # Set torch model url
            checkpoint_path = (
                'https://colorizers.s3.us-east-2.amazonaws.com/'
                'colorization_release_v2-9b330a0b.pth')

        # Set model checkpoint
        if (not checkpoint):
            raise NotImplementedError((
                'Training is not yet supported, `checkpoint` must be '
                'True (bool) for default checkpoint or '
                'a path (str) to a weights file.'
            ))
        else:
            warn = False
            if (isinstance(checkpoint, str)) and (os.path.isfile(checkpoint)):
                self.checkpoint = checkpoint
                warn = True
            else:
                if (warn):
                    warnings.warn(
                        f'Provided checkpoint at {checkpoint} was not found.'
                        f'Resorting to default checkpoint at {checkpoint_path}')

                self.checkpoint = checkpoint_path

        # Build models
        if (self.caffe):
            # Load model from checkpoint
            self.model = self.load_caffe_model(
                prototxt_path, self.checkpoint, points_path)
        else:
            # Set model layers
            self._build_torch_model(norm_layer=norm_layer)
            # Load pretrained state
            if (self.checkpoint.startswith('http')):
                self.pretrained_state = model_zoo.load_url(
                    self.checkpoint, map_location='cpu', check_hash=True,
                    model_dir='weights/zhang_eccv16',
                    file_name='zhang-eccv16-9b330a0b.pth')
            else:
                self.pretrained_state = torch.load(self.checkpoint)

            self.load_state_dict(self.pretrained_state)
            self.eval()


    def load_caffe_model(self, prototxt_path, model_path, points_path):
        """Load serialized Caffe colorization model."""

        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        pts = np.load(points_path)
        # Add the cluster centers as 1x1 convolutions to the model
        class8 = net.getLayerId('class8_ab')
        conv8 = net.getLayerId('conv8_313_rh')
        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype('float32')]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype='float32')]

        return net


    def _build_torch_model(self, norm_layer=nn.BatchNorm2d):
        """Initialize torch model layers."""

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
        self.upsample4 = nn.Upsample(
            scale_factor=4, mode='bicubic', align_corners=False)


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
