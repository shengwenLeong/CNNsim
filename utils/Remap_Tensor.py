#remap the layer which the stride is larger than one to one stride
#
import torch
import math

class Remap_Tensor:
    def __init__(self):

        self.tensor = None
        self.stride = 0

        self.number = 0
        self.channel = 0
        self.height = 0
        self.width = 0

        self.height_ = 0
        self.width_ = 0

        self.pad_h = 0
        self.pad_w = 0

        self.new_height = 0
        self.new_width = 0
        self.new_channel = 0

    def configure(self, tensor, stride):
        self.tensor = tensor
        self.stride = stride

        self.number = tensor.size()[0]
        self.channel = tensor.size()[1]
        self.height = tensor.size()[2]
        self.width = tensor.size()[3]

        self.height_ = math.ceil(self.height / self.stride) * self.stride
        self.width_ = math.ceil(self.width / self.stride) * self.stride

        self.pad_h = self.height_ - self.height
        self.pad_w = self.width_ - self.width

        self.new_height = math.ceil(self.height / self.stride)
        self.new_width = math.ceil(self.width / self.stride)
        self.new_channel = self.channel * self.stride * self.stride

    def pad_tensor(self):
        print(self.tensor.size())
        print(self.number, self.channel, self.height, self.pad_w)
        pad_tensor = self.tensor
        if self.pad_w != 0:
            pad_tensor = torch.cat((self.tensor, torch.zeros(self.number, self.channel, self.height, self.pad_w)), 3)
        if self.pad_h != 0:
            pad_tensor = torch.cat((pad_tensor, torch.zeros(self.number, self.channel, self.pad_h, self.width_)), 2)
        # print(pad_tensor)
        return pad_tensor

    def narrow_tensor(self):

        narrow_height = self.new_height / self.stride
        narrow_width = self.new_width / self.stride
        narrow_tensor = []

        pad_tensor_ = self.pad_tensor()

        for i in range(self.new_height):
            for j in range(self.new_width):
                # print(j,j*self.stride,(j+1)*self.stride)
                # narrow_tensor.append((pad_tensor_.narrow(2,i*self.stride,self.stride)).narrow(3,j*self.stride,self.stride))
                temp = pad_tensor_.narrow(2, i * self.stride, self.stride).narrow(3, j * self.stride, self.stride)
                temp = temp.contiguous().view(self.number, self.new_channel, 1,
                                              1)  # resize(self.number,self.new_channel,1,1)
                narrow_tensor.append(temp)
        return narrow_tensor

    def remap(self):

        remap = self.narrow_tensor()
        # result = []
        # for i in range(len(remap_tensor)):
        #    result.append(remap_tensor[i].resize(self.number,self.new_channel,1,1))
        # result = torch.cat([(torch.cat([remap[j*self.new_width + i] for i in range(self.new_width)],dim=3)) for j in range(self.new_height)],dim=2)
        result = torch.cat([(torch.cat([remap[j * self.new_width + i] for i in range(self.new_width)], dim=3)) for j in
                            range(self.new_height)], dim=2)
        return result

    def return_result(self, tensor, stride):
        self.configure(tensor, stride)
        result = self.remap()
        return result