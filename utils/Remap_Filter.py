# 这个文件用来重排输入权重数据
#
import torch
from queue import Queue
import math

class Remap_Filter:
    # We'll take a parameter for the size of the square arrays to be multiplied, parameter PE size
    def __init__(self, array_size):

        self.array_size = array_size  # the size of PE array
        self.spilt_enable = False  # 判断输入的channel数与PE阵列数目的关系，来决定是否需要切割权重和输入数据，同时判定时钟数和读取结果的时间
        self.sum_cycles = 0

        self.input_number = 0
        self.input_channel = 0
        self.input_height = 0
        self.input_width = 0

        self.kernel_number = 0
        self.kernel_channel = 0
        self.kernel_height = 0
        self.kernel_width = 0

        self.output_channel = 0
        self.output_height = 0
        self.output_width = 0

        # compute window cycle number and all window number at the one layer of DNN

    def configure(self, feature_map, kernel_tensor):
        self.input_number = feature_map.size()[0]
        self.input_channel = feature_map.size()[1]
        self.input_height = feature_map.size()[2]
        self.input_width = feature_map.size()[3]

        self.kernel_number = kernel_tensor.size()[0]
        self.kernel_channel = kernel_tensor.size()[1]
        self.kernel_height = kernel_tensor.size()[2]
        self.kernel_width = kernel_tensor.size()[3]

        self.new_kernel_number = math.ceil(self.kernel_number / self.array_size) * self.array_size
        self.new_kernel_channel = math.ceil(self.kernel_channel / self.array_size) * self.array_size

        self.output_channel = self.new_kernel_number
        self.output_height = self.input_height - self.kernel_height + 1
        self.output_width = self.input_width - self.kernel_width + 1

        if self.input_channel > self.array_size:
            self.sum_cycles = self.new_kernel_channel * self.kernel_height * self.kernel_width
            self.spilt_enable = True
        else:
            self.sum_cycles = self.kernel_channel * self.kernel_height * self.kernel_width
            self.spilt_enable = False
        print('sum_cycles = %d' % self.sum_cycles)

    # Accept a 2d array of weight.
    def fill_weights(self, tensor, weight_input):
        padded_number = self.new_kernel_number - self.kernel_number
        padded_channel = self.new_kernel_channel - self.kernel_channel

        # 补足数据，使其适合PE计算
        pad_tensor = tensor
        if padded_number != 0:
            pad_tensor = torch.cat(
                (tensor, torch.zeros(padded_number, tensor.size()[1], self.kernel_width, self.kernel_height)), 0)
        if self.spilt_enable and padded_channel != 0:
            pad_tensor = torch.cat((pad_tensor, torch.zeros(self.new_kernel_number, padded_channel, self.kernel_width,
                                                            self.kernel_height)), 1)

        print(pad_tensor)
        # 开始push 数据进去队列中，也就是buffer中， 与buffer，不同，这种队列不需要地址的计算，属于FIFO，后续需要地址，则改变
        for k_group in torch.chunk(pad_tensor, int(self.new_kernel_number / self.array_size)):
            # print('----k_group--------')
            # print(k_group)
            for _ in range(int(math.ceil((self.output_height * self.output_width / self.array_size)))):  # 一共重用多少次权重数据
                for kernel_number_group in torch.chunk(k_group, int(k_group.size()[0] / self.array_size), 1):
                    # for kernel_channel_group in torch.chunk(kernel_number_group,int(pad_tensor.size()[1]/array_size),1):
                    # print(kernel_number_group)
                    for y_axis in range(kernel_number_group.size()[2]):
                        for channel_group in range(int(self.new_kernel_channel / self.array_size)):
                            for x_axis in range(kernel_number_group.size()[3]):
                                if self.spilt_enable:
                                    for channel in range(self.array_size * channel_group,
                                                         (self.array_size) * (channel_group + 1)):
                                        print()
                                        for number in range(kernel_number_group.size()[0]):
                                            weight_input[number].put(
                                                kernel_number_group[number][channel][y_axis][x_axis].item())
                                            print('%.4f' % kernel_number_group[number][channel][y_axis][x_axis].item(),
                                                  end=" ")
                                else:
                                    for channel in range(tensor.size()[1]):
                                        print()
                                        for number in range(kernel_number_group.size()[0]):
                                            weight_input[number].put(
                                                kernel_number_group[number][channel][y_axis][x_axis].item())
                                            print('%.4f' % kernel_number_group[number][channel][y_axis][x_axis].item(),
                                                  end=" ")
        '''
        print("FIFO CONTENT")
        number = 0
        while(self.weight_input[number].empty() != 1):
            print()
            for number in range(self.array_size):
                print(self.weight_input[number].get(),end=" ")
        '''

    def fill_activations(self, tensor, activation_input):
        # For the systolic array to function properly, the activations must be
        # padded with a triangle of zeroes
        for row_num in range(self.array_size):
            for _ in range(row_num):
                activation_input[row_num].put(0)

        input_channel = 0
        if self.spilt_enable:
            input_channel = math.ceil(self.input_channel / self.array_size) * self.array_size
        else:
            input_channel = self.input_channel

        padded_channel = math.ceil(self.input_channel / self.array_size) * self.array_size - self.input_channel

        # 补足数据，使其适合PE计算
        pad_tensor = tensor

        if padded_channel != 0 and self.input_channel > self.array_size:
            pad_tensor = torch.cat(
                (pad_tensor, torch.zeros(self.input_number, padded_channel, self.input_width, self.input_height)), 1)
        print('------fill--activation----')

        temp = []
        for i in range(self.output_height):
            for j in range(self.output_width):
                temp_tensor = pad_tensor.narrow(2, i, self.kernel_height).narrow(3, j,
                                                                                 self.kernel_width)  # 按照窗口进行拆分，拆分个数等于输出的map大小
                temp.append(temp_tensor)
        for _ in range(self.output_height * self.output_width % self.array_size):  # 当窗口无法对齐PE阵列时，则补0张量对齐
            temp.append(torch.zeros(1, input_channel, self.kernel_height, self.kernel_width))

        tensor = torch.cat(temp, 0)
        print(tensor.size())
        print(tensor)
        print('-------------------------------------------')
        # 开始push 数据进去队列中，也就是buffer中， 与buffer，不同，这种队列不需要地址的计算，属于FIFO，后续需要地址，则改变
        # feature map 的数据映射以窗口为单位，每隔PE buffer中，存储一个窗口所需要的数据。
        for _ in range(int(math.ceil(self.new_kernel_number / self.array_size))):  # 按照kernel的组数进行存储，即数据重复多少次
            for window_ in range(tensor.size()[0]):  # 窗口 index
                print('--window---%d' % window_)
                if window_ % self.array_size:
                    print()
                for y_axis in range(self.kernel_height):
                    for channel_group in range(int(math.ceil(input_channel / self.array_size))):
                        for x_axis in range(self.kernel_width):
                            if self.spilt_enable:
                                for channel in range(self.array_size * (channel_group),
                                                     (self.array_size) * (channel_group + 1)):
                                    print('%.2f' % tensor[window_][channel][y_axis][x_axis].item(), end=' ')
                                    activation_input[int(window_ % self.array_size)].put(
                                        tensor[window_][channel][y_axis][x_axis].item())
                            else:
                                for channel in range(input_channel):
                                    print('%.2f' % tensor[window_][channel][y_axis][x_axis].item(), end=' ')
                                    activation_input[int(window_ % self.array_size)].put(
                                        tensor[window_][channel][y_axis][x_axis].item())

        print('-------------------------------------------')
        print("Input FIFO CONTENT")
        '''
        number = 0
        while(1):
            if (self.activation_input[0].empty()):
                break;
            print()
            for number in range(self.array_size):
                print('%.2f' % self.activation_input[number].get(),end=" ")
        '''

    @property
    def get_parameters(self):
        return self
    def get_input_info(self):
        return (self.input_number,self.input_channel,self.input_height,self.input_width)

    def get_kernel_info(self):
        return (self.kernel_number,self.kernel_channel,self.kernel_height,self.kernel_width)

    def get_output_info(self):
        return (1,self.output_channel,self.output_height,self.output_width)

    def get_cycle(self):
        return self.cycles
    def get_sum_cycle(self):
        return self.sum_cycles

    def get_activation_input(self):
        return self.activation_input
    def get_weight_input(self):
        return self.weight_input
