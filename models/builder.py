from typing import Dict

from models.layers import UNet2D
from models.layers import UNet3D

from models.models import TPFM
from models.models import TPFM2D


def build_model(config: Dict) -> TPFM:
    loss_type = config.get('loss_type', 'ncc')
    architecture_configs = config.get('architecture')

    backbone = UNet3D(**architecture_configs)

    flownet = TPFM(backbone=backbone, loss_type=loss_type)

    return flownet

def build_model_2d(config: Dict) -> TPFM:
    loss_type = config.get('loss_type', 'ncc')
    architecture_configs = config.get('architecture')
    
    backbone = UNet2D(**architecture_configs)
    
    flownet = TPFM2D(backbone=backbone, loss_type=loss_type)

    return flownet
