"""Implementation of TPFM-DIR model builders from given configs
"""

from typing import Dict

from models.layers import UNet2D, UNet3D

from models.models import TPFM, TPFM2D


def build_model(config: Dict) -> TPFM:
    """Constructs an instance of TPFM for 3D registration

    Args:
        config: A dict containing the backbone architecture configs
                and loss type for training

    Returns:
        TPFM: An instance TPFM-DIR model ready to use
    """
    loss_type = config.get('loss_type', 'ncc')
    architecture_configs = config.get('architecture')

    backbone = UNet3D(**architecture_configs)

    tpfm = TPFM(backbone=backbone, loss_type=loss_type)

    return tpfm

def build_model_2d(config: Dict) -> TPFM2D:
    """Constructs an instance of TPFM for 2D registration

    Args:
        config: A dict containing the backbone architecture configs
                and loss type for training

    Returns:
        TPFM: An instance TPFM-DIR model ready to use
    """
    loss_type = config.get('loss_type', 'ncc')
    architecture_configs = config.get('architecture')
    
    backbone = UNet2D(**architecture_configs)
    
    tpfm = TPFM2D(backbone=backbone, loss_type=loss_type)

    return tpfm
