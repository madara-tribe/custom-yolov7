import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import *
from models.experimental import *


class Yolov7Aux(nn.Module):
    def __init__(self, ch=3):  # model, input channels, number of classes
        super(Yolov7Aux, self).__init__()
        
        #### backbone
        self.ReOrg = ReOrg()
        self.conv1 = Conv(12, 64, k=3, s=1, p=1)
        self.conv2 = Conv(64, 128, k=3, s=2, p=1)
        self.conv3 = Conv(128, 64, k=1, s=1, p=None)
        self.conv4 = Conv(128, 64, k=1, s=1, p=None)
        self.conv5 = Conv(64, 64, k=3, s=1, p=1)
        self.conv6 = Conv(64, 64, k=3, s=1, p=1)
        self.conv7 = Conv(64, 64, k=3, s=1, p=1)
        self.conv8 = Conv(64, 64, k=3, s=1, p=1)
        # 9 cat
        self.conv9 = Conv(256, 128, k=1, s=1, p=None)
        self.conv10 = Conv(128, 256, k=3, s=2, p=1)
        self.conv11 = Conv(256, 128, k=1, s=1, p=None)
        self.conv12 = Conv(256, 128, k=1, s=1, p=None)
        self.conv13 = Conv(128, 128, k=3, s=1, p=1)
        self.conv14 = Conv(128, 128, k=3, s=1, p=1)
        self.conv15 = Conv(128, 128, k=3, s=1, p=1)
        self.conv16 = Conv(128, 128, k=3, s=1, p=1)
        # 19 cat 
        self.conv17 = Conv(512, 256, k=1, s=1, p=None)
        self.conv18 = Conv(256, 512, k=3, s=2, p=1)
        self.conv19 = Conv(512, 256, k=1, s=1, p=None)
        self.conv20 = Conv(512, 256, k=1, s=1, p=None)
        self.conv21 = Conv(256, 256, k=3, s=1, p=1)
        self.conv22 = Conv(256, 256, k=3, s=1, p=1)
        self.conv23 = Conv(256, 256, k=3, s=1, p=1)
        self.conv24 = Conv(256, 256, k=3, s=1, p=1)
        # 27 cat
        self.conv25 = Conv(1024, 512, k=1, s=1, p=None)
        self.conv26 = Conv(512, 768, k=3, s=2, p=1)
        self.conv27 = Conv(768, 384, k=1, s=1, p=None)
        self.conv28 = Conv(768, 384, k=1, s=1, p=None)
        self.conv29 = Conv(384, 384, k=3, s=1, p=1)
        self.conv30 = Conv(384, 384, k=3, s=1, p=1)
        self.conv31 = Conv(384, 384, k=3, s=1, p=1)
        self.conv32 = Conv(384, 384, k=3, s=1, p=1)
        # 36 cat
        self.conv33 = Conv(1536, 768, k=1, s=1, p=None)
        self.conv34 = Conv(768, 1024, k=3, s=2, p=1)
        self.conv35 = Conv(1024, 512, k=1, s=1, p=None)
        self.conv36 = Conv(1024, 512, k=1, s=1, p=None)
        self.conv37 = Conv(512, 512, k=3, s=1, p=1)
        self.conv38 = Conv(512, 512, k=3, s=1, p=1)
        self.conv39 = Conv(512, 512, k=3, s=1, p=1)
        self.conv40 = Conv(512, 512, k=3, s=1, p=1)
        # 45 cat
        self.conv41 = Conv(2048, 1024, k=1, s=1, p=None)

        #### head 
        self.spp1 = SPPCSPC(1024, 512)
        self.conv42 = Conv(512, 384, k=1, s=1, p=None)
        self.up1 = nn.Upsample(scale_factor=2.0, mode='nearest')
        self.conv43 = Conv(768, 384, k=1, s=1, p=None)
        # 51 cat
        self.conv44 = Conv(768, 384, k=1, s=1, p=None)
        self.conv45 = Conv(768, 384, k=1, s=1, p=None)
        self.conv46 = Conv(384, 192, k=3, s=1, p=1)
        self.conv47 = Conv(192, 192, k=3, s=1, p=1)
        self.conv48 = Conv(192, 192, k=3, s=1, p=1)
        self.conv49 = Conv(192, 192, k=3, s=1, p=1)
        # 58 cat
        self.conv50 = Conv(1536, 384, k=1, s=1, p=None)
        self.conv51 = Conv(384, 256, k=1, s=1, p=None)
        self.up2 = nn.Upsample(scale_factor=2.0, mode='nearest')
        self.conv52 = Conv(512, 256, k=1, s=1, p=None)
        # 63 cat
        self.conv53 = Conv(512, 256, k=1, s=1, p=None)
        self.conv54 = Conv(512, 256, k=1, s=1, p=None)
        self.conv55 = Conv(256, 128, k=3, s=1, p=1)
        self.conv56 = Conv(128, 128, k=3, s=1, p=1)
        self.conv57 = Conv(128, 128, k=3, s=1, p=1)
        self.conv58 = Conv(128, 128, k=3, s=1, p=1)
        # 70 cat
        self.conv59 = Conv(1024, 256, k=1, s=1, p=None)
        self.conv60 = Conv(256, 128, k=1, s=1, p=None)
        self.up3 = nn.Upsample(scale_factor=2.0, mode='nearest')
        self.conv61 = Conv(256, 128, k=1, s=1, p=None)
        # 75 cat 
        self.conv62 = Conv(256, 128, k=1, s=1, p=None)
        self.conv63 = Conv(256, 128, k=1, s=1, p=None)
        self.conv64 = Conv(128, 64, k=3, s=1, p=1)
        self.conv65 = Conv(64, 64, k=3, s=1, p=1)
        self.conv66 = Conv(64, 64, k=3, s=1, p=1)
        self.conv67 = Conv(64, 64, k=3, s=1, p=1)
        # 82 cat
        self.conv68 = Conv(512, 128, k=3, s=1, p=None)
        self.conv69 = Conv(128, 256, k=3, s=2, p=1)
        # 85 cat
        self.conv70 = Conv(512, 256, k=1, s=1, p=None)
        self.conv71 = Conv(512, 256, k=1, s=1, p=None)
        self.conv72 = Conv(256, 128, k=3, s=1, p=1)
        self.conv73 = Conv(128, 128, k=3, s=1, p=1)
        self.conv74 = Conv(128, 128, k=3, s=1, p=1)
        self.conv75 = Conv(128, 128, k=3, s=1, p=1)
        # 92 cat
        self.conv76 = Conv(1024, 256, k=1, s=1, p=None)
        self.conv77 = Conv(256, 384, k=3, s=2, p=1)
        # 95 cat
        self.conv78 = Conv(768, 384, k=1, s=1, p=None)
        self.conv79 = Conv(768, 384, k=1, s=1, p=None)
        self.conv80 = Conv(384, 192, k=3, s=1, p=1)
        self.conv81 = Conv(192, 192, k=3, s=1, p=1)
        self.conv82 = Conv(192, 192, k=3, s=1, p=1)
        self.conv83 = Conv(192, 192, k=3, s=1, p=1)
        # 102 cat
        self.conv84 = Conv(1536, 384, k=1, s=1, p=None)
        self.conv85 = Conv(384, 512, k=3, s=2, p=1)
        # 105 cat 
        self.conv86 = Conv(1024, 512, k=1, s=1, p=None)
        self.conv87 = Conv(1024, 512, k=1, s=1, p=None)
        self.conv88 = Conv(512, 256, k=3, s=1, p=1)
        self.conv89 = Conv(256, 256, k=3, s=1, p=1)
        self.conv90 = Conv(256, 256, k=3, s=1, p=1)
        self.conv91 = Conv(256, 256, k=3, s=1, p=1)
        # 112 cat
        self.conv92 = Conv(2048, 512, k=1, s=1, p=None)
        self.conv93 = Conv(128, 256, k=3, s=1, p=1)
        self.conv94 = Conv(256, 512, k=3, s=1, p=1)
        self.conv95 = Conv(384, 768, k=3, s=1, p=1)
        self.conv96 = Conv(512, 1024, k=3, s=1, p=1)
        self.conv97 = Conv(128, 320, k=3, s=1, p=1)
        self.conv98 = Conv(256, 640, k=3, s=1, p=1)
        self.conv99 = Conv(384, 960, k=3, s=1, p=1)
        self.conv100 = Conv(512, 1280, k=3, s=1, p=1)
        #self.detect = IAuxDetect(nc=4, anchors=t["anchors"], ch=[256, 512, 768, 1024, 320, 640, 960, 1280])

    def forward(self, x):
        x1 = self.ReOrg(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(x3)
        x6 = self.conv5(x5)
        x7 = self.conv6(x6)
        x8 = self.conv7(x7)
        x9 = self.conv8(x8)
        x10 = torch.cat((x9, x7, x5, x4), dim=1)

        x11 = self.conv9(x10)
        x12 = self.conv10(x11)
        x13 = self.conv11(x12)
        x14 = self.conv12(x12)
        x15 = self.conv13(x14)
        x16 = self.conv14(x15)
        x17 = self.conv15(x16)
        x18 = self.conv16(x17)
        x19 = torch.cat((x18, x16, x14, x13), dim=1)
        
        x20 = self.conv17(x19)
        x21 = self.conv18(x20)
        x22 = self.conv19(x21)
        x23 = self.conv20(x21)
        x24 = self.conv21(x23)
        x25 = self.conv22(x24)
        x26 = self.conv23(x25)
        x27 = self.conv24(x26)
        x28 = torch.cat((x27, x25, x23, x22), dim=1)

        x29 = self.conv25(x28)
        x30 = self.conv26(x29)
        x31 = self.conv27(x30)
        x32 = self.conv28(x30)
        x33 = self.conv29(x32)
        x34 = self.conv30(x33)
        x35 = self.conv31(x34)
        x36 = self.conv32(x35)
        x37 = torch.cat((x36, x34, x32, x31), dim=1)

        x38 = self.conv33(x37)
        x39 = self.conv34(x38)
        x40 = self.conv35(x39)
        x41 = self.conv36(x39)
        x42 = self.conv37(x41)
        x43 = self.conv38(x42)
        x44 = self.conv39(x43)
        x45 = self.conv40(x44)
        x46 = torch.cat((x45, x43, x41, x40), dim=1)

        x47 = self.conv41(x46)
        x48 = self.spp1(x47)
        x49 = self.conv42(x48)
        x50 = self.up1(x49)
        x51 = self.conv43(x38)
        x52 = torch.cat((x51, x50), dim=1)

        x53 = self.conv44(x52)
        x54 = self.conv45(x52)
        x55 = self.conv46(x54)
        x56 = self.conv47(x55)
        x57 = self.conv48(x56)
        x58 = self.conv49(x57)
        x59 = torch.cat((x58, x57, x56, x55, x54, x53), dim=1)

        x60 = self.conv50(x59)
        x61 = self.conv51(x60)
        x62 = self.up2(x61)
        x63 = self.conv52(x29)
        x64 = torch.cat((x63, x62), dim=1)

        x65 = self.conv53(x64)
        x66 = self.conv54(x64)
        x67 = self.conv55(x66)
        x68 = self.conv56(x67)
        x69 = self.conv57(x68)
        x70 = self.conv58(x69)
        x71 = torch.cat((x70, x69, x68, x67, x66, x65), dim=1)

        x72 = self.conv59(x71)
        x73 = self.conv60(x72)
        x74 = self.up3(x73)
        x75 = self.conv61(x20)
        x76 = torch.cat((x75, x74), dim=1)

        x77 = self.conv62(x76)
        x78 = self.conv63(x76)
        x79 = self.conv64(x78)
        x80 = self.conv65(x79)
        x81 = self.conv66(x80)
        x82 = self.conv67(x81)
        x83 = torch.cat((x82, x81, x80, x79, x78, x77), dim=1)

        x84 = self.conv68(x83)
        x85 = self.conv69(x84)
        x86 = torch.cat((x85, x72), dim=1)

        x87 = self.conv70(x86)
        x88 = self.conv71(x86)
        x89 = self.conv72(x88)
        x90 = self.conv73(x89)
        x91 = self.conv74(x90)
        x92 = self.conv75(x91)
        x93 = torch.cat((x92, x91, x90, x89, x88, x87), dim=1)

        x94 = self.conv76(x93)
        x95 = self.conv77(x94)
        x96 = torch.cat((x95, x60), dim=1)

        x97 = self.conv78(x96)
        x98 = self.conv79(x96)
        x99 = self.conv80(x98)
        x100 = self.conv81(x99)
        x101 = self.conv82(x100)
        x102 = self.conv83(x101)
        x103 = torch.cat((x102, x101, x100, x99, x98, x97), dim=1)

        x104 = self.conv84(x103)
        x105 = self.conv85(x104) 
        x106 = torch.cat((x105, x48), dim=1)

        x107 = self.conv86(x106)
        x108 = self.conv87(x106)
        x109 = self.conv88(x108)
        x110 = self.conv89(x109)
        x111 = self.conv90(x110)
        x112 = self.conv91(x111)
        x113 = torch.cat((x112, x111, x110, x109, x108, x107), dim=1)

        x114 = self.conv92(x113)
        x115 = self.conv93(x84)
        x116 = self.conv94(x94)
        x117 = self.conv95(x104)
        x118 = self.conv96(x114)
        x119 = self.conv97(x84)
        x120 = self.conv98(x72)
        x121 = self.conv99(x60)
        x122 = self.conv100(x48)
        return [x115, x116, x117, x118, x119, x120, x121, x122]
    

 

#if __name__ == '__main__':
 #   from torchsummary import summary
 #   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 #   model = Yolov7Aux().to(device)
 #   summary(model, (3, 640, 640))
 ##   img = torch.rand(1, 3, 640, 640).to(device)
 #   y = model(img)
 #   print(len(y))
    

