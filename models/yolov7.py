import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from models.common import *
from models.experimental import *

# torch.Size([1, 3, 80, 80, 9]) torch.Size([1, 3, 40, 40, 9]) torch.Size([1, 3, 20, 20, 9]) 3
# torch.Size([1, 3, 80, 80, 9]) torch.Size([1, 3, 40, 40, 9]) torch.Size([1, 3, 20, 20, 9]) 3
#cfg ='cfg/training/yolov7.yaml'
#with open(cfg) as f:
#    t = yaml.load(f, Loader=yaml.SafeLoader)
    
class RepConvs_(nn.Module):
    def __init__(self):
        super(RepConvs_, self).__init__()
        self.repconv1 = RepConv(128, 256)
        self.repconv2 = RepConv(256, 512)
        self.repconv3 = RepConv(512, 1024)
    def forward(self, x):
        x1 = self.repconv1(x[0])
        x2 = self.repconv2(x[1])
        x3 = self.repconv3(x[2])
        return [x1, x2, x3]
        
class Yolov7(nn.Module):
    def __init__(self, ch=3, yaml_file=None):  # model, input channels, number of classes
        super(Yolov7, self).__init__()
        t = yaml_file
        self.conv1 = Conv(ch, 32, k=3, p=1)
        self.conv2 = Conv(32, 64, k=3, s=2, p=1)
        self.conv3 = Conv(64, 64, k=3, p=1)
        self.conv4 = Conv(64, 128, k=3, s=2, p=1)
        self.conv4 = Conv(64, 128, k=3, s=2, p=1)
        self.conv5_1 = Conv(128, 64, k=1, p=None)
        self.conv5_2 = Conv(128, 64, k=1, p=None)
        self.conv6 = Conv(64, 64, k=3, p=1)
        self.conv7 = Conv(64, 64, k=3, p=1)
        self.conv8 = Conv(64, 64, k=3, p=1)
        self.conv9 = Conv(64, 64, k=3, p=1)
        # 10 cat

        self.conv11 = Conv(256, 256, k=1, p=None)
        self.mp1 = MP(k=2)
        self.conv13 = Conv(256, 128, k=1, p=None)
        self.conv14 = Conv(256, 128, k=1, p=None)
        self.conv15 = Conv(128, 128, k=3, s=2, p=1)
        # 16 cat 

        self.conv17 = Conv(256, 128, k=1, s=1, p=None)
        self.conv18 = Conv(256, 128, k=1, s=1, p=None)
        self.conv19 = Conv(128, 128, k=3, s=1, p=1)
        self.conv20 = Conv(128, 128, k=3, s=1, p=1)
        self.conv21 = Conv(128, 128, k=3, s=1, p=1)
        self.conv22 = Conv(128, 128, k=3, s=1, p=1)
        # 23 cat

        self.conv24 = Conv(512, 512, k=1, s=1, p=None)
        self.mp2 = MP(k=2)
        self.conv26 = Conv(512, 256, k=1, s=1, p=None)
        self.conv27 = Conv(512, 256, k=1, s=1, p=None)
        self.conv28 = Conv(256, 256, k=3, s=2, p=1)
        # 29 cat

        self.conv30 = Conv(512, 256, k=1, s=1, p=None)
        self.conv31 = Conv(512, 256, k=1, s=1, p=None)
        self.conv32 = Conv(256, 256, k=3, s=1, p=1)
        self.conv33 = Conv(256, 256, k=3, s=1, p=1)
        self.conv34 = Conv(256, 256, k=3, s=1, p=1)
        self.conv35 = Conv(256, 256, k=3, s=1, p=1)

        # 35 cat
        self.conv36 = Conv(1024, 1024, k=1, s=1, p=None)
        self.mp3 = MP(k=2)
        self.conv37 = Conv(1024, 512, k=1, s=1, p=None)
        self.conv38 = Conv(1024, 512, k=1, s=1, p=None)
        self.conv39 = Conv(512, 512, k=3, s=2, p=1)
        # cat

        self.conv40 = Conv(1024, 256, k=1, s=1, p=None)
        self.conv41 = Conv(1024, 256, k=1, s=1, p=None)
        self.conv42 = Conv(256, 256, k=3, s=1, p=1)
        self.conv43 = Conv(256, 256, k=3, s=1, p=1)
        self.conv44 = Conv(256, 256, k=3, s=1, p=1)
        self.conv45 = Conv(256, 256, k=3, s=1, p=1)

        # cat
        self.conv46 = Conv(1024, 1024, k=1, s=1, p=None)


        # head
        self.spp = SPPCSPC(1024, 512)
        self.conv47 = Conv(512, 256, k=1, p=None)
        self.up1 = nn.Upsample(scale_factor=2.0, mode='nearest')
        self.conv48 = Conv(1024, 256, k=1, p=None)
        # cat
        self.conv49 = Conv(512, 256, k=1, p=None)
        self.conv50 = Conv(512, 256, k=1, p=None)
        self.conv51 = Conv(256, 128, k=3, s=1, p=1)
        self.conv52 = Conv(128, 128, k=3, s=1, p=1)
        self.conv53 = Conv(128, 128, k=3, s=1, p=1)
        self.conv54 = Conv(128, 128, k=3, s=1, p=1)
        # cat

        self.conv55 = Conv(1024, 256, k=1, s=1, p=None)
        self.conv56 = Conv(256, 128, k=1, s=1, p=None)
        self.up2 = nn.Upsample(scale_factor=2.0, mode='nearest')
        self.conv57 = Conv(512, 128, k=1, s=1, p=None)
        # cat

        self.conv58 = Conv(256, 128, k=1, s=1, p=None)
        self.conv59 = Conv(256, 128, k=1, s=1, p=None)
        self.conv60 = Conv(128, 64, k=3, s=1, p=1)
        self.conv61 = Conv(64, 64, k=3, s=1, p=1)
        self.conv62 = Conv(64, 64, k=3, s=1, p=1)
        self.conv63 = Conv(64, 64, k=3, s=1, p=1)
        # cat

        self.conv64 = Conv(512, 128, k=1, s=1, p=None)
        self.mp4 = MP(k=2)
        self.conv65 = Conv(128, 128, k=1, s=1, p=None)
        self.conv66 = Conv(128, 128, k=1, s=1, p=None)
        self.conv67 = Conv(128, 128, k=3, s=2, p=1)
        # cat

        self.conv68 = Conv(512, 256, k=1, s=1, p=None)
        self.conv69 = Conv(512, 256, k=1, s=1, p=None)
        self.conv70 = Conv(256, 128, k=3, s=1, p=1)
        self.conv71 = Conv(128, 128, k=3, s=1, p=1)
        self.conv72 = Conv(128, 128, k=3, s=1, p=1)
        self.conv73 = Conv(128, 128, k=3, s=1, p=1)
        # cat

        self.conv74 = Conv(1024, 256, k=1, s=1, p=None)
        self.mp5 = MP(k=2)
        self.conv75 = Conv(256, 256, k=1, s=1, p=None)
        self.conv76 = Conv(256, 256, k=1, s=1, p=None)
        self.conv77 = Conv(256, 256, k=3, s=2, p=1)
        # cat

        self.conv78 = Conv(1024, 512, k=1, s=1, p=None)
        self.conv79 = Conv(1024, 512, k=1, s=1, p=None)
        self.conv80 = Conv(512, 256, k=3, s=1, p=1)
        self.conv81 = Conv(256, 256, k=3, s=1, p=1)
        self.conv82 = Conv(256, 256, k=3, s=1, p=1)
        self.conv83 = Conv(256, 256, k=3, s=1, p=1)
        # cat

        self.conv84 = Conv(2048, 512, k=1, s=1, p=None)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5_ = self.conv5_1(x4)
        x5 = self.conv5_2(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv8(x8)
        x10 = torch.cat((x9, x7, x5, x5_), dim=1)

        # 11
        x11 = self.conv11(x10)
        x12 = self.mp1(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x10)
        x15 = self.conv15(x14)
        x16 = torch.cat((x13, x15), dim=1)
        
        x17 = self.conv17(x16)
        x18 = self.conv18(x16)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        x21 = self.conv21(x20)
        x22 = self.conv22(x21)
        x23 = torch.cat((x22, x20, x18, x17), dim=1)

        x24 = self.conv24(x23)
        x25 = self.mp2(x24)
        x26 = self.conv26(x25)
        x27 = self.conv27(x23)
        x28 = self.conv28(x27)
        x29 = torch.cat((x26, x28), dim=1)

        x30 = self.conv30(x29)
        x31 = self.conv31(x29)
        x32 = self.conv32(x31)
        x33 = self.conv33(x32)
        x34 = self.conv34(x33)
        x35 = self.conv35(x34)
        #print("x35", x35.shape)
        x36 = torch.cat((x35, x33, x31, x30), dim=1)

        x37 = self.conv36(x36)
        x38 = self.mp3(x37)
        x39 = self.conv37(x38)
        x40 = self.conv38(x37)
        x41 = self.conv39(x40)
        x42 = torch.cat((x41, x39), dim=1)

        x43 = self.conv40(x42)
        x44 = self.conv41(x42)
        x45 = self.conv42(x44)
        x46 = self.conv43(x45)
        x47 = self.conv44(x46)
        x48 = self.conv45(x47)
        x49 = torch.cat((x48, x46, x44, x43), dim=1)
        x50 = self.conv46(x49)

        # head 
        x51 = self.spp(x50)
        x52 = self.conv47(x51)
        x53 = self.up1(x52)
        x54 = self.conv48(x36)
        x55 = torch.cat((x53, x54), dim=1)

        x56 = self.conv49(x55)
        x57 = self.conv50(x55)
        x58 = self.conv51(x57)
        x59 = self.conv52(x58)
        x60 = self.conv53(x59)
        x61 = self.conv54(x60)
        x62 = torch.cat((x61, x60, x59, x58, x57, x56), dim=1)

        x63 = self.conv55(x62)
        x64 = self.conv56(x63)
        x65 = self.up2(x64)
        x66 = self.conv57(x23)
        x67 = torch.cat((x66, x65), dim=1)

        x68 = self.conv58(x67)
        x69 = self.conv59(x67)
        x70 = self.conv60(x69)
        x71 = self.conv61(x70)
        x72 = self.conv62(x71)
        x73 = self.conv63(x72)
        x74 = torch.cat((x73, x72, x71, x70, x69, x68), dim=1)

        x75 = self.conv64(x74)
        x76 = self.mp4(x75)
        x77 = self.conv65(x76)
        x78 = self.conv66(x75)
        x79 = self.conv67(x78)
        x80 = torch.cat((x79, x77, x63), dim=1)

        x81 = self.conv68(x80)
        x82 = self.conv69(x80)
        x83 = self.conv70(x82)
        x84 = self.conv71(x83)
        x85 = self.conv72(x84)
        x86 = self.conv73(x85)
        x87 = torch.cat((x86, x85, x84, x83, x82, x81), dim=1)

        x88 = self.conv74(x87)
        x89 = self.mp5(x88)
        x90 = self.conv75(x89)
        x91 = self.conv76(x88)
        x92 = self.conv77(x91)
        x93 = torch.cat((x92, x90, x51), dim=1)

        x94 = self.conv78(x93)
        x95 = self.conv79(x93)
        x96 = self.conv80(x95)
        x97 = self.conv81(x96)
        x98 = self.conv82(x97)
        x99 = self.conv83(x98)
        x100 = torch.cat((x99, x98, x97, x96, x95, x94), dim=1)

        x101 = self.conv84(x100)
        #x102 = self.repconv1(x75)
        #x103 = self.repconv2(x88)
        #x104 = self.repconv3(x101)
        #x105 = self.detect([x102, x103, x104])
        return x75, x88, x101
    
  
if __name__ == '__main__':
    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Yolov8().to(device)
    #summary(model, (3, 640, 640))
    img = torch.rand(1, 3, 640, 640).to(device)
    y = model(img)
    print(y[0].shape)
    #print(model)


