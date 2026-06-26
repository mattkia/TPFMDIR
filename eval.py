import yaml
import torch
import argparse

from trainers import DiceTester
from trainers import TRETester


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config',
                    type=str,
                    default='oasis',
                    help='JSON file for configuration')

args = parser.parse_args()


torch.cuda.empty_cache()
with open(f'configs/{args.config}.yaml', 'r') as handle:
    config = yaml.safe_load(handle)

if config.get('data')['name'] in ['oasis', 'lpba40', 'ixi', 'candi', 'mindboggle', 'abdomen']:
    tester = DiceTester(config)
elif config.get('data')['name'] in ['lungct']:
    tester = TRETester(config)

tester.run()