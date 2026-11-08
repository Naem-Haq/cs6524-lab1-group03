
import yaml

from PIL import Image
from os import path as opath
from mypylibs import figures, filters, fourier, trajectory

ROOT_IN = 'in'
ROOT_OUT = 'out'
EXT = 'png'

def main(case: int):

    img = Image.open(opath.join(ROOT_IN, f'{case:02}.{EXT}'))
    params = {}
    config = opath.join(ROOT_IN, f'{case:02}.yaml')
    if opath.exists(config):
        with open(config) as f:
            params = yaml.safe_load(f) or {}
    num_samples = params.pop('num_samples', 500)

    boundary, steps = filters.extract_boundary(img, **params)
    x, y = trajectory.calculate(boundary, num_samples)
    descriptors = fourier.Descriptors(x, y)

    figures.save_transforms(steps, opath.join(ROOT_OUT, f'{case:02}_transforms.{EXT}'))
    figures.save_trajectory(descriptors.z, opath.join(ROOT_OUT, f'{case:02}_trajectory.{EXT}'))
    figures.save_final(descriptors, opath.join(ROOT_OUT, f'{case:02}_final.{EXT}'))

if __name__ == '__main__':

    case = 0
    while opath.exists(opath.join(ROOT_IN, f'{case:02}.{EXT}')):
        print(f'Processing case {case:02}')
        main(case)
        case += 1
