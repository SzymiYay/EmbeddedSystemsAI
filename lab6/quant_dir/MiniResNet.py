# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class MiniResNet(torch.nn.Module):
    def __init__(self):
        super(MiniResNet, self).__init__()
        self.module_0 = py_nndct.nn.Input() #MiniResNet::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MiniResNet::MiniResNet/Sequential[CNN]/Conv2d[0]/input.2
        self.module_2 = py_nndct.nn.ReLU(inplace=False) #MiniResNet::MiniResNet/Sequential[CNN]/ReLU[1]/input.3
        self.module_3 = py_nndct.nn.Conv2d(in_channels=16, out_channels=4, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[2]/Sequential[L1]/Conv2d[0]/input.4
        self.module_5 = py_nndct.nn.ReLU(inplace=False) #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[2]/Sequential[L1]/ReLU[2]/input.6
        self.module_6 = py_nndct.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[2]/Sequential[L2]/Conv2d[0]/input.7
        self.module_8 = py_nndct.nn.ReLU(inplace=False) #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[2]/Sequential[L2]/ReLU[2]/117
        self.module_9 = py_nndct.nn.Add() #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[2]/input.9
        self.module_10 = py_nndct.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MiniResNet::MiniResNet/Sequential[CNN]/Conv2d[3]/input.10
        self.module_11 = py_nndct.nn.ReLU(inplace=False) #MiniResNet::MiniResNet/Sequential[CNN]/ReLU[4]/130
        self.module_12 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #MiniResNet::MiniResNet/Sequential[CNN]/MaxPool2d[5]/input.11
        self.module_13 = py_nndct.nn.Conv2d(in_channels=32, out_channels=4, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[6]/Sequential[L1]/Conv2d[0]/input.12
        self.module_15 = py_nndct.nn.ReLU(inplace=False) #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[6]/Sequential[L1]/ReLU[2]/input.14
        self.module_16 = py_nndct.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[6]/Sequential[L2]/Conv2d[0]/input.15
        self.module_18 = py_nndct.nn.ReLU(inplace=False) #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[6]/Sequential[L2]/ReLU[2]/170
        self.module_19 = py_nndct.nn.Add() #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[6]/input.17
        self.module_20 = py_nndct.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[7]/Sequential[L1]/Conv2d[0]/input.18
        self.module_22 = py_nndct.nn.ReLU(inplace=False) #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[7]/Sequential[L1]/ReLU[2]/input.20
        self.module_23 = py_nndct.nn.Conv2d(in_channels=2, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[7]/Sequential[L2]/Conv2d[0]/input.21
        self.module_25 = py_nndct.nn.ReLU(inplace=False) #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[7]/Sequential[L2]/ReLU[2]/206
        self.module_26 = py_nndct.nn.Add() #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[7]/input.23
        self.module_27 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MiniResNet::MiniResNet/Sequential[CNN]/Conv2d[8]/input.24
        self.module_28 = py_nndct.nn.ReLU(inplace=False) #MiniResNet::MiniResNet/Sequential[CNN]/ReLU[9]/219
        self.module_29 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #MiniResNet::MiniResNet/Sequential[CNN]/MaxPool2d[10]/input.25
        self.module_30 = py_nndct.nn.Conv2d(in_channels=64, out_channels=8, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[11]/Sequential[L1]/Conv2d[0]/input.26
        self.module_32 = py_nndct.nn.ReLU(inplace=False) #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[11]/Sequential[L1]/ReLU[2]/input.28
        self.module_33 = py_nndct.nn.Conv2d(in_channels=8, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[11]/Sequential[L2]/Conv2d[0]/input.29
        self.module_35 = py_nndct.nn.ReLU(inplace=False) #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[11]/Sequential[L2]/ReLU[2]/259
        self.module_36 = py_nndct.nn.Add() #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[11]/input.31
        self.module_37 = py_nndct.nn.Conv2d(in_channels=64, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[12]/Sequential[L1]/Conv2d[0]/input.32
        self.module_39 = py_nndct.nn.ReLU(inplace=False) #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[12]/Sequential[L1]/ReLU[2]/input.34
        self.module_40 = py_nndct.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[12]/Sequential[L2]/Conv2d[0]/input.35
        self.module_42 = py_nndct.nn.ReLU(inplace=False) #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[12]/Sequential[L2]/ReLU[2]/295
        self.module_43 = py_nndct.nn.Add() #MiniResNet::MiniResNet/Sequential[CNN]/ResidualBlock[12]/input.37
        self.module_44 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MiniResNet::MiniResNet/Sequential[CNN]/Conv2d[13]/input.38
        self.module_45 = py_nndct.nn.ReLU(inplace=False) #MiniResNet::MiniResNet/Sequential[CNN]/ReLU[14]/input.39
        self.module_46 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MiniResNet::MiniResNet/Sequential[CNN]/Conv2d[15]/input.40
        self.module_47 = py_nndct.nn.ReLU(inplace=False) #MiniResNet::MiniResNet/Sequential[CNN]/ReLU[16]/319
        self.module_48 = py_nndct.nn.Module('flatten') #MiniResNet::MiniResNet/Sequential[FC]/Flatten[0]/input.41
        self.module_49 = py_nndct.nn.Linear(in_features=1152, out_features=10, bias=True) #MiniResNet::MiniResNet/Sequential[FC]/Linear[1]/input
        self.module_50 = py_nndct.nn.Module('softmax',dim=1) #MiniResNet::MiniResNet/Sequential[FC]/Softmax[2]/329

    def forward(self, *args):
        self.output_module_0 = self.module_0(input=args[0])
        self.output_module_1 = self.module_1(self.output_module_0)
        self.output_module_2 = self.module_2(self.output_module_1)
        self.output_module_3 = self.module_3(self.output_module_2)
        self.output_module_5 = self.module_5(self.output_module_3)
        self.output_module_6 = self.module_6(self.output_module_5)
        self.output_module_8 = self.module_8(self.output_module_6)
        self.output_module_9 = self.module_9(alpha=1, input=self.output_module_2, other=self.output_module_8)
        self.output_module_10 = self.module_10(self.output_module_9)
        self.output_module_11 = self.module_11(self.output_module_10)
        self.output_module_12 = self.module_12(self.output_module_11)
        self.output_module_13 = self.module_13(self.output_module_12)
        self.output_module_15 = self.module_15(self.output_module_13)
        self.output_module_16 = self.module_16(self.output_module_15)
        self.output_module_18 = self.module_18(self.output_module_16)
        self.output_module_19 = self.module_19(alpha=1, input=self.output_module_12, other=self.output_module_18)
        self.output_module_20 = self.module_20(self.output_module_19)
        self.output_module_22 = self.module_22(self.output_module_20)
        self.output_module_23 = self.module_23(self.output_module_22)
        self.output_module_25 = self.module_25(self.output_module_23)
        self.output_module_26 = self.module_26(alpha=1, input=self.output_module_19, other=self.output_module_25)
        self.output_module_27 = self.module_27(self.output_module_26)
        self.output_module_28 = self.module_28(self.output_module_27)
        self.output_module_29 = self.module_29(self.output_module_28)
        self.output_module_30 = self.module_30(self.output_module_29)
        self.output_module_32 = self.module_32(self.output_module_30)
        self.output_module_33 = self.module_33(self.output_module_32)
        self.output_module_35 = self.module_35(self.output_module_33)
        self.output_module_36 = self.module_36(alpha=1, input=self.output_module_29, other=self.output_module_35)
        self.output_module_37 = self.module_37(self.output_module_36)
        self.output_module_39 = self.module_39(self.output_module_37)
        self.output_module_40 = self.module_40(self.output_module_39)
        self.output_module_42 = self.module_42(self.output_module_40)
        self.output_module_43 = self.module_43(alpha=1, input=self.output_module_36, other=self.output_module_42)
        self.output_module_44 = self.module_44(self.output_module_43)
        self.output_module_45 = self.module_45(self.output_module_44)
        self.output_module_46 = self.module_46(self.output_module_45)
        self.output_module_47 = self.module_47(self.output_module_46)
        self.output_module_48 = self.module_48(start_dim=1, input=self.output_module_47, end_dim=3)
        self.output_module_49 = self.module_49(self.output_module_48)
        self.output_module_50 = self.module_50(self.output_module_49)
        return self.output_module_50
