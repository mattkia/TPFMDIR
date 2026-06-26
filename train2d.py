import yaml
import torch
import argparse

from trainers2d import DiceTrainer


if __name__ == '__main__':
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
        trainer = DiceTrainer(config)
    else:
        raise NotImplementedError()

    trainer.run()