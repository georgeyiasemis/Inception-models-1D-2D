import torch
import torch.nn as nn
from pytorch_model_summary import summary


# Inception Model/ GoogLeNet 2D
class Conv2d_Block(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2d_Block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.batchnorm(out)
        out = self.relu(out)
        return out

class Inception2d_Block(nn.Module):

    def __init__(self, in_channels, out_1x1, in_reduce_3x3, out_reduce_3x3,
                    in_reduce_5x5, out_reduce_5x5, out_1x1_maxpool):
        """Short summary.

        Parameters
        ----------
        in_channels : type
            Description of parameter `in_channels`.
        out_1x1 : type
            Description of parameter `out_1x1`.
        in_reduce_3x3 : type
            Description of parameter `in_reduce_3x3`.
        out_reduce_3x3 : type
            Description of parameter `out_reduce_3x3`.
        in_reduce_5x5 : type
            Description of parameter `in_reduce_5x5`.
        out_reduce_5x5 : type
            Description of parameter `out_reduce_5x5`.
        out_1x1_maxpool : type
            Description of parameter `out_1x1_maxpool`.

        Returns
        -------
        type
            Description of returned object.

        """
        super(Inception2d_Block, self).__init__()

        self.branch1 = Conv2d_Block(in_channels=in_channels, out_channels=out_1x1, kernel_size=(1,1))

        self.branch2 = nn.Sequential(
            Conv2d_Block(in_channels=in_channels, out_channels=in_reduce_3x3, kernel_size=(1,1)),
            Conv2d_Block(in_channels=in_reduce_3x3, out_channels=out_reduce_3x3, kernel_size=(3,3), stride=(1,1), padding=1)
        )
        self.branch3 = nn.Sequential(
            Conv2d_Block(in_channels=in_channels, out_channels=in_reduce_5x5, kernel_size=(1,1)),
            Conv2d_Block(in_channels=in_reduce_5x5, out_channels=out_reduce_5x5, kernel_size=(5,5), padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv2d_Block(in_channels=in_channels, out_channels=out_1x1_maxpool, kernel_size=(1,1))
        )

    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        return out

class GoogLeNet(nn.Module):

    def __init__(self, in_channels=3, out_dim=1000):
        super(GoogLeNet, self).__init__()

        # 1
        self.conv1 = Conv2d_Block(in_channels=in_channels, out_channels=64, kernel_size=(7,7),
            stride=(2,2), padding=3)

        # 2
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)
        self.conv2 = Conv2d_Block(in_channels=64, out_channels=192, kernel_size=(3,3),
            stride=(1,1), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)

        # 3
        self.inception3a = Inception2d_Block(in_channels=192, out_1x1=64, in_reduce_3x3=96, out_reduce_3x3=128,
                        in_reduce_5x5=16, out_reduce_5x5=32, out_1x1_maxpool=32)
        self.inception3b = Inception2d_Block(in_channels=256, out_1x1=128, in_reduce_3x3=128, out_reduce_3x3=192,
                        in_reduce_5x5=32, out_reduce_5x5=96, out_1x1_maxpool=64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)

        # 4
        self.inception4a = Inception2d_Block(in_channels=480, out_1x1=192, in_reduce_3x3=96, out_reduce_3x3=208,
                        in_reduce_5x5=16, out_reduce_5x5=48, out_1x1_maxpool=64)
        self.inception4b = Inception2d_Block(in_channels=512, out_1x1=160, in_reduce_3x3=112, out_reduce_3x3=224,
                        in_reduce_5x5=24, out_reduce_5x5=64, out_1x1_maxpool=64)
        self.inception4c = Inception2d_Block(in_channels=512, out_1x1=128, in_reduce_3x3=128, out_reduce_3x3=256,
                        in_reduce_5x5=24, out_reduce_5x5=64, out_1x1_maxpool=64)
        self.inception4d = Inception2d_Block(in_channels=512, out_1x1=112, in_reduce_3x3=144, out_reduce_3x3=288,
                        in_reduce_5x5=32, out_reduce_5x5=64, out_1x1_maxpool=64)
        self.inception4e = Inception2d_Block(in_channels=528, out_1x1=256, in_reduce_3x3=160, out_reduce_3x3=320,
                        in_reduce_5x5=32, out_reduce_5x5=128, out_1x1_maxpool=128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)

        # 5
        self.inception5a = Inception2d_Block(in_channels=832, out_1x1=256, in_reduce_3x3=160, out_reduce_3x3=320,
                        in_reduce_5x5=32, out_reduce_5x5=128, out_1x1_maxpool=128)
        self.inception5b = Inception2d_Block(in_channels=832, out_1x1=384, in_reduce_3x3=192, out_reduce_3x3=384,
                        in_reduce_5x5=48, out_reduce_5x5=128, out_1x1_maxpool=128)

        # 6
        self.avgpool = nn.AvgPool2d(kernel_size=(7,7), stride=(1,1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, out_dim)

    def forward(self, x):

        out = self.conv1(x)

        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.maxpool2(out)

        out = self.inception3a(out)
        out = self.inception3b(out)
        out = self.maxpool3(out)

        out = self.inception4a(out)
        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)
        out = self.inception4e(out)
        out = self.maxpool4(out)

        out = self.inception5a(out)
        out = self.inception5b(out)

        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.dropout(out)
        out = self.fc1(out)

        return out

#######################################

# Inception Model/ GoogLeNet 1D
class Conv1d_Block(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv1d_Block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.batchnorm(out)
        out = self.relu(out)
        return out

class Inception1d_Block(nn.Module):

    def __init__(self, in_channels, out_1x1, in_reduce_3x3, out_reduce_3x3,
                    in_reduce_5x5, out_reduce_5x5, out_1x1_maxpool):
        """Short summary.

        Parameters
        ----------
        in_channels : type
            Description of parameter `in_channels`.
        out_1x1 : type
            Description of parameter `out_1x1`.
        in_reduce_3x3 : type
            Description of parameter `in_reduce_3x3`.
        out_reduce_3x3 : type
            Description of parameter `out_reduce_3x3`.
        in_reduce_5x5 : type
            Description of parameter `in_reduce_5x5`.
        out_reduce_5x5 : type
            Description of parameter `out_reduce_5x5`.
        out_1x1_maxpool : type
            Description of parameter `out_1x1_maxpool`.

        Returns
        -------
        type
            Description of returned object.

        """
        super(Inception1d_Block, self).__init__()

        self.branch1 = Conv1d_Block(in_channels=in_channels, out_channels=out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            Conv1d_Block(in_channels=in_channels, out_channels=in_reduce_3x3, kernel_size=1),
            Conv1d_Block(in_channels=in_reduce_3x3, out_channels=out_reduce_3x3, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            Conv1d_Block(in_channels=in_channels, out_channels=in_reduce_5x5, kernel_size=1),
            Conv1d_Block(in_channels=in_reduce_5x5, out_channels=out_reduce_5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            Conv1d_Block(in_channels=in_channels, out_channels=out_1x1_maxpool, kernel_size=1)
        )

    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        return out

class AuxInceptionNet(nn.Module):

    def __init__(self, in_channels, out_dim):
        super(AuxInceptionNet, self).__init__()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.7)
        self.avgpool = nn.AvgPool1d(kernel_size=5, stride=3)
        self.conv = Conv1d_Block(in_channels=in_channels, out_channels=512, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, out_dim)

    def forward(self, x):

        out = self.avgpool(x)
        out = self.conv(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)

        return out



class GoogLeNet1D(nn.Module):

    def __init__(self, in_channels, out_dim, aux_loss=True):
        super(GoogLeNet1D, self).__init__()

        self.aux_loss = aux_loss
        if aux_loss:
            self.auxnet1 = AuxInceptionNet(in_channels=512, out_dim=out_dim)
            self.auxnet2 = AuxInceptionNet(in_channels=528, out_dim=out_dim)

        # 1
        self.conv1 = Conv1d_Block(in_channels=in_channels, out_channels=64, kernel_size=7,
            stride=2, padding=3)

        # 2
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv1d_Block(in_channels=64, out_channels=192, kernel_size=3,
            stride=1, padding=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 3
        self.inception3a = Inception1d_Block(in_channels=192, out_1x1=64, in_reduce_3x3=96, out_reduce_3x3=128,
                        in_reduce_5x5=16, out_reduce_5x5=32, out_1x1_maxpool=32)
        self.inception3b = Inception1d_Block(in_channels=256, out_1x1=128, in_reduce_3x3=128, out_reduce_3x3=192,
                        in_reduce_5x5=32, out_reduce_5x5=96, out_1x1_maxpool=64)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 4
        self.inception4a = Inception1d_Block(in_channels=480, out_1x1=192, in_reduce_3x3=96, out_reduce_3x3=208,
                        in_reduce_5x5=16, out_reduce_5x5=48, out_1x1_maxpool=64)
        self.inception4b = Inception1d_Block(in_channels=512, out_1x1=160, in_reduce_3x3=112, out_reduce_3x3=224,
                        in_reduce_5x5=24, out_reduce_5x5=64, out_1x1_maxpool=64)
        self.inception4c = Inception1d_Block(in_channels=512, out_1x1=128, in_reduce_3x3=128, out_reduce_3x3=256,
                        in_reduce_5x5=24, out_reduce_5x5=64, out_1x1_maxpool=64)
        self.inception4d = Inception1d_Block(in_channels=512, out_1x1=112, in_reduce_3x3=144, out_reduce_3x3=288,
                        in_reduce_5x5=32, out_reduce_5x5=64, out_1x1_maxpool=64)
        self.inception4e = Inception1d_Block(in_channels=528, out_1x1=256, in_reduce_3x3=160, out_reduce_3x3=320,
                        in_reduce_5x5=32, out_reduce_5x5=128, out_1x1_maxpool=128)
        self.maxpool4 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 5
        self.inception5a = Inception1d_Block(in_channels=832, out_1x1=256, in_reduce_3x3=160, out_reduce_3x3=320,
                        in_reduce_5x5=32, out_reduce_5x5=128, out_1x1_maxpool=128)
        self.inception5b = Inception1d_Block(in_channels=832, out_1x1=384, in_reduce_3x3=192, out_reduce_3x3=384,
                        in_reduce_5x5=48, out_reduce_5x5=128, out_1x1_maxpool=128)

        # 6
        self.avgpool = nn.AvgPool1d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, out_dim)

    def forward(self, x):

        out = self.conv1(x)

        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.maxpool2(out)

        out = self.inception3a(out)
        out = self.inception3b(out)
        out = self.maxpool3(out)

        out = self.inception4a(out)

        if self.aux_loss and self.training:
            aux_out1 = self.auxnet1(out)

        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)

        if self.aux_loss and self.training:
            aux_out2 = self.auxnet2(out)

        out = self.inception4e(out)
        out = self.maxpool4(out)

        out = self.inception5a(out)
        out = self.inception5b(out)

        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.dropout(out)
        out = self.fc1(out)

        if self.aux_loss and self.training:
            return aux_out1, aux_out2, out
        else:
            return out

#######################################
if __name__ == '__main__':

    model = GoogLeNet1D(10, 1)
    x = torch.randn(10, 10, 224)
    # model.training = True
    print(model(x)[0].shape,model(x)[1].shape, model.training)
    print(summary(model, x , show_hierarchical=False, show_input=False, show_parent_layers=True))
#     m = Conv2d_Block(3, 16, kernel_size=(2,1), padding=(2,1))
#     print(m(x).shape)
