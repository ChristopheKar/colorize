import torch
from torch import nn
from torch.utils import model_zoo

from .base import Base


class SIGGRAPH17Model(Base):

    def __init__(self, norm_layer=nn.BatchNorm2d, classes=529, pretrained=None):
        super().__init__()

        # Conv1
        self.model1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(64)
        )
        # Conv2
        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128)
        )
        # Conv3
        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256)
        )
        # Conv4
        self.model4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512)
        )
        # Conv5
        self.model5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512)
        )
        # Conv6
        self.model6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512)
        )
        # Conv7
        self.model7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512)
        )
        # Conv8
        self.model8up = nn.Sequential(nn.ConvTranspose2d(
            512, 256, kernel_size=4, stride=2, padding=1, bias=True))
        self.model3short8 = nn.Sequential(nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1, bias=True))

        self.model8 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256)
        )
        # Conv9
        self.model9up = nn.Sequential(nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1, bias=True))
        self.model2short9 = nn.Sequential(nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=True))
        # add the two feature maps above

        self.model9 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128))

        # Conv10
        self.model10up = nn.Sequential(nn.ConvTranspose2d(
            128, 128, kernel_size=4, stride=2, padding=1, bias=True))
        self.model1short10 = nn.Sequential(nn.Conv2d(
            64, 128, kernel_size=3, stride=1, padding=1, bias=True))

        # Add the two feature maps above
        self.model10 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=.2)
        )

        # Classification output
        self.model_class = nn.Sequential(nn.Conv2d(
            256, classes, kernel_size=1,
            padding=0, dilation=1, stride=1, bias=True))

        # Regression output
        self.model_out = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=True),
            nn.Tanh()
        )

        self.upsample4 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear'))
        self.softmax = nn.Sequential(nn.Softmax(dim=1))

        # Load pretrained state
        self._validate_keys(pretrained, 'model', argname='pretrained')
        if (pretrained['model'].startswith('http')):
            self.pretrained_state = model_zoo.load_url(
                pretrained['model'], map_location='cpu', check_hash=True)
        else:
            self.pretrained_state = torch.load(pretrained['model'])

        self.load_state_dict(self.pretrained_state)
        self.eval()


    def forward(self, input_A, input_B=None, mask_B=None):
        if (input_B is None):
            input_B = torch.cat((input_A*0, input_A*0), dim=1)
        if (mask_B is None):
            mask_B = input_A*0

        input_tensor = torch.cat(
            (self.normalize_l(input_A), self.normalize_ab(input_B), mask_B),
            dim=1)
        conv1_2 = self.model1(input_tensor)
        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
        conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)

        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)
        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_3 = self.model9(conv9_up)
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_2 = self.model10(conv10_up)
        out_reg = self.model_out(conv10_2)

        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_3 = self.model9(conv9_up)
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_2 = self.model10(conv10_up)
        out_reg = self.model_out(conv10_2)

        return self.denormalize_ab(out_reg)


if __name__ == '__main__':
    # Checkpoint available at
    # https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth
    checkpoint = dict(model='weights/siggraph17-df00044c.pth')
    model = SIGGRAPH17Model(pretrained=checkpoint)
