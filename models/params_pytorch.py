import torchvision.models as models
import torch
import argparse
from ptflops import get_model_complexity_info
from models.SCanNet.SCanNet import SCanNet

from models.DTCDSCN.DTCDSCN import CDNet34
from models.Changeformer.ChangeFormer import ChangeFormerV6
from models.DMINet.DMINet import DMINet
from models.AERNet.network import AERNet
from models.AMTNet.AMTNet import CDNet
from models.UABCD.UABCD import UABCD
from models.MDIPNet.MDIPNet import build_model
from models.HAFF.modelMCD import ADVNets
from models.WHFCE import WaveHFD, BaseNet

with torch.cuda.device(0):
  # net = models.resnet18()

  # 变化检测领域

  # YG2
  # net = YG2get_segmentation_model("YG2")

  # U-Net
  net = UABCD(6,2)

  flops, params = get_model_complexity_info(net, (3, 256, 256), as_strings=True, print_per_layer_stat=True) #不用写batch_size大小，默认batch_size=1
  print('Flops:  ' + flops)
  print('Params: ' + params)
