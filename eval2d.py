import yaml
import torch
import argparse

from trainers2d import DiceTester


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config',
                    type=str,
                    default='acdc_unet',
                    help='JSON file for configuration')

args = parser.parse_args()


torch.cuda.empty_cache()
with open(f'configs/{args.config}.yaml', 'r') as handle:
    config = yaml.safe_load(handle)

if config.get('data')['name'] in ['acdc', 'camus']:
    tester = DiceTester(config)
else:
    raise NotImplementedError()

tester.run()