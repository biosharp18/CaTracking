import os
import numpy as np
import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt

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



        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        #h_next = torch.sigmoid(h_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv1.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv1.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            #print(f"Forward for layer {layer_idx}")
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len): # loop over sequence
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

def train_epoch(model, train_X, train_Y, batch_size, time_batch_size):
    model.train()
    epoch_loss = 0
    for i in range(0,train_X.shape[0], batch_size):
        #print(f"FREE MEMORY: {torch.cuda.mem_get_info()[0] / 2**30} GiB")

        ##This is the problem here
        batch = train_X[i:min(train_X.shape[0],i+batch_size),:,:,:,:].float()
        true_batch = train_Y[i:min(train_Y.shape[0],i+batch_size),:,:,:,:].float()
        #output = model(batch) #is list of [output of all layers, hidden states of all layers]

        b, _, _, h, w = batch.size()

        init_states = []
        for i in range(model.num_layers):
            init_states.append((train_Y[i*batch_size:(i+1)*batch_size,0,:,:,:].to(device), train_Y[i*batch_size:(i+1)*batch_size,0,:,:,:].to(device)))
        hidden_state = init_states



        seq_len = batch.size(1)
        cur_layer_input = batch.clone()
        for time_batch in range(0,seq_len, time_batch_size):
            layer_output_list = []
            last_state_list = []
            for layer_idx in range(model.num_layers):
                #print(f"Forward for layer {layer_idx}")
                h, c = hidden_state[layer_idx]
                output_inner = []
                for t in range(time_batch, min(seq_len,time_batch + time_batch_size)): # loop over sequence
                    #print(t)
                    #print(cur_layer_input.shape)
                    h, c = model.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :].to(device),
                                                    cur_state=[h, c])
                    output_inner.append(h)
                layer_output = torch.stack(output_inner, dim=1)
                #cur_layer_input = layer_output ##DISABLES MULTI LAYER INFERENCE

                layer_output_list.append(layer_output)
                last_state_list.append([h, c])

                output_all_layers = layer_output_list[:]
                output_all_hidden = last_state_list[:]
                output_last_layer = output_all_layers[-1]
                time_batch_loss = loss_fun(output_last_layer, true_batch[i*batch_size:(i+1)*batch_size,time_batch:min(seq_len,time_batch + time_batch_size),:,:,:].to(device))
        #print(output_last_layer.shape)
        #print(true_batch.shape)
                epoch_loss += time_batch_loss
                optimizer.zero_grad()
                time_batch_loss.backward()
                optimizer.step()
    return epoch_loss

@torch.no_grad()
def val_epoch(model, val_X, val_Y, batch_size, time_batch_size):
    model.eval()
    epoch_loss = 0
    for i in range(0,val_X.shape[0], batch_size):
        #print(f"FREE MEMORY: {torch.cuda.mem_get_info()[0] / 2**30} GiB")

        ##This is the problem here
        batch = val_X[i:min(val_X.shape[0],i+batch_size),:,:,:,:].float()
        true_batch = val_Y[i:min(val_Y.shape[0],i+batch_size),:,:,:,:].float()
        #output = model(batch) #is list of [output of all layers, hidden states of all layers]

        b, _, _, h, w = batch.size()

        init_states = []
        for i in range(model.num_layers):
            init_states.append((val_Y[i*batch_size:(i+1)*batch_size,0,:,:,:].to(device), val_Y[i*batch_size:(i+1)*batch_size,0,:,:,:].to(device)))
        hidden_state = init_states



        seq_len = batch.size(1)
        cur_layer_input = batch.clone()
        for time_batch in range(0,seq_len, time_batch_size):
            layer_output_list = []
            last_state_list = []
            for layer_idx in range(model.num_layers):
                #print(f"Forward for layer {layer_idx}")
                h, c = hidden_state[layer_idx]
                output_inner = []
                for t in range(time_batch, min(seq_len,time_batch + time_batch_size)): # loop over sequence
                    #print(t)
                    #print(cur_layer_input.shape)
                    h, c = model.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :].to(device),
                                                    cur_state=[h, c])
                    output_inner.append(h)
                layer_output = torch.stack(output_inner, dim=1)
                #cur_layer_input = layer_output ##DISABLES MULTI LAYER INFERENCE

                layer_output_list.append(layer_output)
                last_state_list.append([h, c])

                output_all_layers = layer_output_list[:]
                output_all_hidden = last_state_list[:]
                output_last_layer = output_all_layers[-1]
                time_batch_loss = loss_fun(output_last_layer, true_batch[i*batch_size:(i+1)*batch_size,time_batch:min(seq_len,time_batch + time_batch_size),:,:,:].to(device))
        #print(output_last_layer.shape)
        #print(true_batch.shape)
                epoch_loss += time_batch_loss
    return epoch_loss

@torch.no_grad()
def my_forward(model, time_batch_size, X,Y):
    with torch.no_grad():
        b, t, _, h, w = X.size()

        init_states = []
        for i in range(model.num_layers):
            init_states.append((Y[i:(i+1),0,:,:,:].to(device), Y[i:(i+1),0,:,:,:].to(device)))
        hidden_state = init_states



        seq_len = t
        cur_layer_input = X.clone()
        layer_output_list = []
        last_state_list = []
        for time_batch in range(0,seq_len, time_batch_size):

            for layer_idx in range(model.num_layers):
                #print(f"Forward for layer {layer_idx}")
                h, c = hidden_state[layer_idx]
                output_inner = []
                for t in range(time_batch, min(seq_len,time_batch + time_batch_size)): # loop over sequence
                    #print(t)
                    #print(cur_layer_input.shape)
                    h, c = model.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :].to(device),
                                                    cur_state=[h, c])
                    output_inner.append(h)
                layer_output = torch.stack(output_inner, dim=1)
                #cur_layer_input = layer_output ##DISABLES MULTI LAYER INFERENCE

                layer_output_list.append(layer_output)
                last_state_list.append([h, c])

                output_all_layers = layer_output_list[:]
                output_all_hidden = last_state_list[:]
                output_last_layer = output_all_layers[-1]
        for time_batch in range(1, len(output_all_layers)):
            output_all_layers[0] = torch.cat((output_all_layers[0], output_all_layers[time_batch]), dim = 1)

    return output_all_layers[0], output_all_hidden

def train(model, nepochs, batch_size, time_batch_size, scheduler, optimizer, loss_fun, train_X, train_Y, val_X, val_Y):
    #train_X will be np array(batch, time, channels, height, width)
    b_train, t_train, c, h, w = train_X.shape
    loss_function = loss_fun
    train_loss = np.zeros((nepochs))
    val_loss = np.zeros((nepochs))
    for epoch in range(nepochs):
        for phase in ["train","val"]:
            if phase == "train":
                epoch_loss = train_epoch(model, train_X, train_Y, batch_size, time_batch_size)
                train_loss[epoch] = epoch_loss
                print(f"EPOCH{epoch}, train loss: {epoch_loss}")
            if phase =="val":
                epoch_loss = val_epoch(model, val_X, val_Y, batch_size, time_batch_size)
                val_loss[epoch] = epoch_loss
                print(f"EPOCH{epoch}, val loss: {epoch_loss}")

        print(f"FREE MEMORY: {torch.cuda.mem_get_info()[0] / 2**30} GiB")

    fig, ax = plt.subplots(nrows = 1, ncols = 2)
    ax[0].set_title("Train Loss")
    ax[0].plot(train_loss)
    ax[1].set_title("Validation Loss")
    ax[1].plot(val_loss)
    plt.show()

def load_images(raw_folder, mask_folder):
    total_imgs = len(os.listdir(mask_folder))
    masked_list = os.listdir(mask_folder)
    original_list = os.listdir(raw_folder)
    img_shape = np.asarray(Image.open(raw_folder + "/"+original_list[0]))[:,:,:2].transpose((2,0,1)).shape
    #Use train/val split of 70/30
    train_ind = round(0.7*total_imgs)

    train_X = np.zeros((1,train_ind,img_shape[0],(img_shape[1]//4 + 1)*4,(img_shape[2]//4 + 1)*4))
    train_Y = np.zeros((1,train_ind,1,(img_shape[1]//4 + 1)*4,(img_shape[2]//4 + 1)*4))
    test_X = np.zeros((1,total_imgs - train_ind,img_shape[0],(img_shape[1]//4 + 1)*4,(img_shape[2]//4 + 1)*4))
    test_Y = np.zeros((1,total_imgs - train_ind,1,(img_shape[1]//4 + 1)*4,(img_shape[2]//4 + 1)*4))
    print(img_shape, train_X.shape)

    for file in range(train_ind):
        train_file = np.asarray(Image.open(raw_folder + "/"+original_list[file]))[:,:,:2].transpose((2,0,1))
        mask_file = np.asarray(Image.open(mask_folder + "/"+masked_list[file]))[:,:,:1].transpose((2,0,1))
        train_file = (train_file - train_file.mean()) / train_file.std()
        mask_file = (mask_file -mask_file.mean()) / mask_file.std()
        #Input sizes must be divisible by 4 for U NET. Pad images.
        train_file = np.pad(train_file,pad_width= ((0,0),(0,(train_file.shape[1]//4 + 1)*4 - train_file.shape[1]),(0,0)))
        mask_file = np.pad(mask_file,pad_width= ((0,0),(0,(mask_file.shape[1]//4 + 1)*4 - mask_file.shape[1]),(0,0)))

        train_file = np.pad(train_file,pad_width= ((0,0),(0,0),(0,(train_file.shape[2]//4 + 1)*4 - train_file.shape[2])))
        mask_file = np.pad(mask_file,pad_width= ((0,0),(0,0),(0,(mask_file.shape[2]//4 + 1)*4 - mask_file.shape[2])))

        train_X[0,file,:,:,:] = train_file
        train_Y[0,file,:,:,:] = mask_file
    for file in range(train_ind, total_imgs):
        test_file = np.asarray(Image.open(raw_folder + "/"+original_list[file]))[:,:,:2].transpose((2,0,1))
        mask_file = np.asarray(Image.open(mask_folder + "/"+masked_list[file]))[:,:,:1].transpose((2,0,1))
        test_file = (test_file -test_file.mean()) / test_file.std()
        mask_file = (mask_file -mask_file.mean()) / mask_file.std()

        test_file = np.pad(test_file,pad_width= ((0,0),(0,(test_file.shape[1]//4 + 1)*4 - test_file.shape[1]),(0,0)))
        mask_file = np.pad(mask_file,pad_width= ((0,0),(0,(mask_file.shape[1]//4 + 1)*4 - mask_file.shape[1]),(0,0)))

        test_file = np.pad(test_file,pad_width= ((0,0),(0,0),(0,(test_file.shape[2]//4 + 1)*4 - test_file.shape[2])))
        mask_file = np.pad(mask_file,pad_width= ((0,0),(0,0),(0,(mask_file.shape[2]//4 + 1)*4 - mask_file.shape[2])))

        test_X[0,file - train_ind,:,:,:] = test_file
        test_Y[0,file - train_ind,:,:,:] = mask_file

    #Convert to tensors
    train_X = torch.from_numpy(train_X.astype(np.float32))
    train_Y = torch.from_numpy(train_Y.astype(np.float32))
    test_X = torch.from_numpy(test_X.astype(np.float32))
    test_Y = torch.from_numpy(test_Y.astype(np.float32))

    return train_X, train_Y, test_X, test_Y




device = "cuda" if torch.cuda.is_available() else "cpu"

loss_fun = torch.nn.BCEWithLogitsLoss()
#Model params are how much? 25769148416 bytes??
free_mem = torch.cuda.mem_get_info()[0]
print(f"FREE GPU MEMORY: {free_mem}")
C = ConvLSTM(input_dim = 2, hidden_dim = 1, kernel_size = (3,3), num_layers = 1, batch_first = True, return_all_layers = True).to(device)
model_sz = (free_mem - torch.cuda.mem_get_info()[0]) / 2**30
print(f"MODEL SIZE ON GPU {model_sz} GiB")

optimizer = torch.optim.Adam(C.parameters(),lr = 0.01) #Try Adam optimizer, because large plateau


train_vids = [11408, 11409]
test_vids = [11409]
for dataset in range(1):
    train_X, train_Y, test_X, test_Y = load_images(f"./data/original/original/{train_vids[dataset]}", f"./data/masked/masked/{train_vids[dataset]}")
    train(C, nepochs = 50, batch_size = 1, time_batch_size = 30, scheduler = None, optimizer = optimizer,loss_fun = loss_fun, train_X = train_X, train_Y = train_Y, val_X = test_X , val_Y = test_Y)



#Visualize predictions
with torch.no_grad():
    test_frames = 100
    C.eval()
    _, _, test_X, test_Y = load_images(f"./data/original/original/{test_vids[0]}", f"./data/masked/masked/{test_vids[0]}")
    preds, hidden = my_forward(C, 30, test_X[0:1,0:test_frames,:,:,:].float().to(device), test_Y[0:1,0:1,:,:,:].float().to(device))
    preds = preds.detach().cpu().numpy()
    print(preds.shape)

    for i in range(0,test_frames):
        fig, ax = plt.subplots(nrows = 1, ncols = 3)
        ax[0].set_title("prediction")
        ax[0].imshow(preds[0,i,0,:,:])
        ax[1].set_title("ground truth")
        ax[1].imshow(test_Y[0,i,0,:,:].numpy())
        ax[2].set_title("original")
        ax[2].imshow(test_X[0,i,0,:,:].numpy())
        plt.savefig(f"pred{i}.png")
        plt.close()