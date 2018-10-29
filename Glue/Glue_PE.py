# 这个文件用于将计算单元组织起来，形成一个加速器的结构
#

import compute_unit as c_unit
from queue import Queue
import math
import utils

class SystolicArray:
    # We'll take a parameter for the size of the square arrays to be multiplied, parameter PE size
    def __init__(self, array_size):

        self.array_size = array_size  # the size of PE array
        # The inputs and outputs will both be FIFO queues
        self.cycles = 0
        self.Remap_Filter = None
        self.activation_input = [Queue() for _ in range(self.array_size)]
        self.weight_input = [Queue() for _ in range(self.array_size)]

        # "cells" will hold the array of processing elements  //the all dimensions of PE array
        self.cells = []
        # This array is a square with dimensions "array_size"
        for _ in range(self.array_size):
            row = []
            for _ in range(self.array_size):
                cell = c_unit.SystolicArrayCell()  # the one dimensions of PE array
                row.append(cell)
            self.cells.append(row)

        self.output_enable = Queue()
        self.output = [Queue() for _ in range(self.array_size)]

        # When all cells and inputs are created, then they can be connected
        # (again, this would be accomplished with wiring)
        for row_num, row in enumerate(self.cells):
            for col_num, cell in enumerate(row):
                cell.connect(col_num, row_num, self)


    def clear_register(self):
        for row in self.cells:
            for cell in row:
                cell.clear_register()

    # For this demo, all cells will read() the values of their neighbors first
    def read(self):
        for row in self.cells:
            for cell in row:
                cell.read()

    # And then after all cells have read(), they will compute() the next step
    def compute(self):
        for row in self.cells:
            for cell in row:
                cell.compute()

    def output_result(self):
        for row in range(self.array_size):
            for column in range(self.array_size):
                print(self.cells[row][column].partial_sum_out)
                self.output[row].put(self.cells[row][column].partial_sum_out)

    def output_result_(self, column):
        for row in range(self.array_size):
            print(self.cells[row][column].partial_sum_out)
            self.output[row].put(self.cells[row][column].partial_sum_out)
        for row in range(self.array_size):
            self.cells[row][column].clear_register()

    def output_result_enable(self, window_number):
        print('self.cycles %d' % self.cycles)
        print(window_number)
        print(((self.Remap_Filter.get_sum_cycle() * window_number) - 1))
        print(self.cycles == (self.Remap_Filter.get_sum_cycle() * window_number - 1))
        if (self.cycles == (self.Remap_Filter.get_sum_cycle() * window_number - 1)):
            print("output_enable")
            for i in range(self.array_size):
                self.output_enable.put(i)

    # Each cycle involves a read() and a compute()
    def cycle(self, window_number):

        print('cycle %d' % self.cycles)
        # read() models register sampling on the positive edge of the clock
        self.read()
        # compute() models the combinational logic between clock edges
        self.compute()

        self.output_result_enable(window_number)

        if not self.output_enable.empty():
            self.output_result_(self.output_enable.get())
        self.cycles = self.cycles + 1

    # run() will execute the array's computation, assuming it's been filled
    def run(self):
        # It takes 3n-2 cycles to compute the full matrix of results
        temp = 0
        counter = 0
        output_channel = math.ceil(self.Remap_Filter.get_kernel_info()[0] / self.array_size)
        width_number = math.ceil(self.Remap_Filter.get_input_info()[3] / self.array_size)
        height_number = math.ceil(self.Remap_Filter.get_output_info()[2] * self.Remap_Filter.get_output_info()[3] / self.array_size)

        print("----hotizaontal_number----")
        print(output_channel, width_number, height_number)
        print("---------------------------")

        for _ in range(output_channel):
            for h in range(height_number):
                # for w in range(width_number):
                #    counter = counter + 1
                counter = counter + 1
                for k in range(self.Remap_Filter.get_sum_cycle()):
                    print()
                    print('---------------')
                    self.cycle(counter)
        for _ in range(self.array_size):
            self.cycle(counter)
        print('----result-------')
        return self.get_outputs()

    def execute(self, input_map, kernel, stride):
        remap_ = utils.Remap_Tensor()
        r_kernel = remap_.return_result(kernel, stride)
        r_input_map = remap_.return_result(input_map , stride)
        self.Remap_Filter = utils.Remap_Filter(self.array_size)
        self.Remap_Filter.configure(r_input_map,r_kernel)
        self.Remap_Filter.fill_weights(r_kernel,self.weight_input)
        self.Remap_Filter.fill_activations(r_input_map,self.activation_input)
        result = self.run()
        return result

    # The outputs are also staggered and transposed, so we'll format them
    # before returning the results
    def get_outputs(self):
        ret = []
        number = 0
        while (1):
            if (self.output[0].empty() and self.output[1].empty()):
                break;
            print()
            for i in range(self.array_size):
                print('%.4f' % self.output[i].get(), end=" ")
        # the results
        for row_num in range(self.array_size):
            row = []
            for output_col in self.output:
                row.append(output_col.get())
            ret.append(row)

        return ret