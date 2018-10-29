# this is file is designed for compute engine
#
#
from queue import Queue
from decimal import Decimal

class SystolicArrayCell:
    def __init__(self):
        # For connection purposes, the cell needs to be able to find its
        # neighbors in the array. In real hardware, this would be done with
        # wiring and the X and Y position wouldn't matter
        self.pos_x = 0
        self.pos_y = 0

        # Each cell has the concept of a "partial sum" and an "activation".
        # These take one cycle to cross each cell (they would be delayed
        # with a register). To model this in python, we'll have a <field>
        # variable that represents the value driven by the neighboring cell,
        # and a <field>_out value representing the value driven by this cell.

        # partial sum: the running sum of the products, which always reside in PE
        self.partial_sum = 0
        self.partial_sum_out = 0
        # activation: the input activation value, transmitted horizontally
        self.activation = 0
        self.activation_out = 0

        # weight: The weight representing the input weight valur transmitted horizontally from left to right

        self.weight = 0
        self.weight_out = 0

        # Input fields, which will hold the connection to the cells or FIFOs
        # above and to the left of this cell
        self.input_weight = None

        # Input fields, which will hold the connection to the cells or FIFOs
        # above and to the left of this cell
        self.input_activation = None
        self.input_partial_sum = None

    # In the hardware implementation, we would use a control flow signal and
    # weight inputs via the partial sum lines (note that a weight is only half
    # the bits of that field, allowing control flow to be transmitted
    # alongside). For simplification here, we'll just say it's hacked in by
    # magic.
    def clear_register(self):
        self.partial_sum = 0;
        self.partial_sum_out = 0;

    # Connects this cell to its neighbors above and to the left
    def connect(self, pos_x, pos_y, array):
        self.pos_x = pos_x
        self.pos_y = pos_y

        # If we're at x position zero, then our left neighbor is a FIFO queue
        if self.pos_x is 0:
            self.input_weight = array.weight_input[self.pos_y]
        # Otherwise, it's another cell
        else:
            self.input_weight = array.cells[self.pos_y][self.pos_x - 1]

        # If we're at y position zero, then our above neighbor is a FIFO queue
        if self.pos_y is 0:
            self.input_activation = array.activation_input[self.pos_x]
        # Otherwise, it's another cell
        else:
            self.input_activation = array.cells[0][self.pos_x]

    # We'll model the transfer of signals through registers with a read() and a
    # compute() method.
    # read() represents the registers sampling data at the positive edge of the
    # clock
    def read(self):
        # Read the left neighbor for weight
        # If this is a FIFO queue, take its value (or 0 if it's empty)
        if type(self.input_weight) is Queue:
            if self.input_weight.empty():
                # print("queue empty")
                self.weight = Decimal('0').quantize(Decimal('0.00'))
            else:
                # print("queue no empty")
                self.weight = Decimal(self.input_weight.get()).quantize(Decimal('0.0000'))
        # If it is a cell, we read the value from activation_out
        else:
            # print("no queue")
            self.weight = Decimal(self.input_weight.weight_out).quantize(Decimal('0.0000'))
        # print(self.weight)

        # Read the left neighbor for feature map
        # If this is a FIFO queue, take its value (or 0 if it's empty)
        if type(self.input_activation) is Queue:
            if self.input_activation.empty():
                self.activation = Decimal('0').quantize(Decimal('0.00'))
                # self.activation_out = self.activation
            else:
                self.activation = Decimal(self.input_activation.get()).quantize(Decimal('0.00'))
                # self.activation_out = self.activation
        else:
            self.activation = Decimal(self.input_activation.activation_out).quantize(Decimal('0.00'))
            # self.activation_out = self.activation
        self.activation_out = self.activation

    # compute() represents combinational logic that takes place between
    # positive edges of the clock (multiplication and addition)
    def compute(self):
        # First, the weight and activation in are multiplied
        if self.pos_x == 0:
            print()
        print(self.weight, "*", self.activation, end=' | ')

        # print(self.activation)
        # print("--------------")
        product = self.weight * self.activation
        # print(product)
        # print("==============")
        # Then that value is added to the partial sum from above and transmitted
        # downwards
        self.partial_sum_out = self.partial_sum_out + product

        # if self.pos_x == 0:
        #    print()
        # print(self.partial_sum_out , end=' | ')
        # And the weight is transmitted to the right
        self.weight_out = self.weight
