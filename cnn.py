import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.activation_fun = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels= 64,
                              kernel_size=self.kernel_size,
                              padding="same",
                              bias=self.bias)
        self.pool1 = nn.MaxPool2d(2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128,kernel_size = self.kernel_size, padding = "same")
        self.pool2 = nn.MaxPool2d((2,2),stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256,kernel_size = self.kernel_size, padding = "same")



        self.upsample1 = nn.Upsample(scale_factor = 2, mode = 'bilinear')
        self.deconv1 = nn.Conv2d(in_channels = 256, out_channels = 128,kernel_size= self.kernel_size, stride = 1,padding = "same")
        self.deconv1_2 = nn.Conv2d(in_channels = 256, out_channels = 128,kernel_size = self.kernel_size, stride = 1,padding = "same") #after concat
        self.upsample2 = nn.Upsample(scale_factor = 2, mode = 'bilinear')
        self.deconv2 = nn.Conv2d(in_channels = 128, out_channels = 64,kernel_size= self.kernel_size, stride = 1,padding = "same")
        self.deconv2_2 = nn.Conv2d(in_channels = 128, out_channels = 64,kernel_size = self.kernel_size, stride = 1,padding = "same") #after concat
        self.finalconv = nn.Conv2d(in_channels = 64, out_channels = 4*self.hidden_dim, kernel_size = self.kernel_size, padding = "same")


    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        #print("combined", combined.shape)
        layer1 = self.conv1(combined)
        layer1 = self.activation_fun(layer1)
        #print("layer1 shape", layer1.shape)
        layer2 = self.pool1(layer1)
        layer2 = self.conv2(layer2)
        layer2 = self.activation_fun(layer2)
        #print("layer2 shape", layer2.shape)
        layer3 = self.pool2(layer2)
        layer3 = self.conv3(layer3)
        layer3 = self.activation_fun(layer3)
        #print("layer3 shape", layer3.shape)
        layer4 = self.upsample1(layer3)
        layer4 = self.deconv1(layer4)
        #print(layer4.shape)
        layer4 = self.deconv1_2(torch.cat((layer2, layer4),dim = 1))
        layer5 = self.upsample2(layer4)
        layer5 = self.deconv2(layer5)
        layer5 = self.deconv2_2(torch.cat((layer1, layer5),dim = 1))
        combined_conv = self.finalconv(layer5)