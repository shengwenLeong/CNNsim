import Glue
import torch
import torch.nn.functional as F

def main():
    myArray = Glue.SystolicArray(2)
    conv_filter = torch.tensor([[[[-1, -2],
                                  [-3, -4]],

                                 [[-5, -6],
                                  [-7, -8]],

                                 [[-9, -10],
                                  [-11, -12]],

                                 [[-13, -14],
                                  [-15, -16]]],

                                [[[-17, -18],
                                  [-19, -20]],

                                 [[-21, -22],
                                  [-23, -24]],

                                 [[-25, -26],
                                  [-27, -28]],

                                 [[-29, -30],
                                  [-31, -32]]],

                                [[[-33, -34],
                                  [-35, -36]],

                                 [[-37, -38],
                                  [-39, -40]],

                                 [[-41, -42],
                                  [-43, -44]],

                                 [[-45, -46],
                                  [-47, -48]]]], dtype=torch.float)
    input_feature_map = torch.tensor([[[[1.1, 1.2, 1.3, 1.4],
                                        [1.5, 1.6, 1.7, 1.8],
                                        [1.9, 2.0, 2.1, 2.2],
                                        [2.3, 2.4, 2.5, 2.6]],

                                       [[2.7, 2.8, 2.9, 3.0],
                                        [3.1, 3.2, 3.3, 3.4],
                                        [3.5, 3.6, 3.7, 3.8],
                                        [3.9, 4.0, 4.1, 4.2]],

                                       [[4.3, 4.4, 4.5, 4.6],
                                        [4.7, 4.8, 4.9, 5.0],
                                        [5.1, 5.2, 5.3, 5.4],
                                        [5.5, 5.6, 5.7, 5.8]],

                                       [[5.9, 6.0, 6.1, 6.2],
                                        [6.3, 6.4, 6.5, 6.6],
                                        [6.7, 6.8, 6.9, 7.0],
                                        [7.1, 7.2, 7.3, 7.4]]]], dtype=torch.float)
    # conv_filter = torch.rand(3,3,3,3)
    # input_feature_map = torch.rand(1,3,5,5)
    result = F.conv2d(input_feature_map, conv_filter, padding=0, stride=1)
    print(result)
    print('-------------------next--------------------')
    res = myArray.execute(input_feature_map, conv_filter, 1)
if __name__ == '__main__':
    main()

