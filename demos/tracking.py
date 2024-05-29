import torch
import numpy as np
import os
from monai.transforms import LoadImaged, Compose, ScaleIntensityRanged
from monai.data import ITKReader
from src.transforms.transforms import CreatePatchGrid, GEMTransform, ResamplePatch
from src.tracking.iterative import CenterlineTracker
from src.models.gemcnn import GEMGCN
from src.utils.general import transform_points
import matplotlib.pyplot as plt

from pdb import set_trace as bp

def main(img, seedpoint, **tracking_params):
    transform = get_transforms(**tracking_params)
    # pre-process image data
    sample = transform({'img': img})

    # create SIRE sampler
    SIRE_sampler = ResamplePatch(
        tracking_params['TransformParams']['n_points'],
        p=0.,
        mode=tracking_params['TransformParams']['mode']
        )
    
    # create model + load weights
    if tracking_params['TransformParams']['mode'] == 'spherical':
        network = GEMGCN(
            nlayers=len(tracking_params['TransformParams']['scales']),
            nverts=642,
            convs=3,
            channels=16
        )
    network.load_state_dict(torch.load(tracking_params['NetworkParams']['state_dict']))
    network.cuda()

    # transform seedpoint to voxel coordinates
    seedpoint = transform_points(seedpoint.reshape(1,3),
                                 torch.linalg.inv(sample['img_meta_dict']['affine'])).squeeze()
    
    # UNCOMMENT TO VISUALIZE SEEDPOINT
    # plt.imshow(sample['img'][int(seedpoint[2]),:,:], 'gray')
    # plt.plot(seedpoint[0], seedpoint[1], 'r.')
    # plt.savefig('seedpoint_test.png')

    tracker = CenterlineTracker(network,
                                SIRE_sampler,
                                stepsize=tracking_params['Tracking']['stepsize'],
                                criterion=tracking_params['Tracking']['criterion']
                                )
    
    path = tracker(seedpoint, sample)
    path = torch.stack(path)
    print(path.shape)


def get_transforms(**tracking_params):
    # transforms needed to load and normalize CTA data

    # adapt this part if needed
    loader = LoadImaged(keys=['img'],
                            image_only=False)
    loader.register(ITKReader(reverse_indexing=True, affine_lps_to_ras=False))

    transform = [loader,
                 ScaleIntensityRanged(keys=['img'],
                 a_min=-400,
                 a_max=800,
                 b_min=0,
                 b_max=1,
                 clip=False
                 )]
    
    # add SIRE transforms
    transform += [CreatePatchGrid(
        tracking_params['TransformParams']['scales'],
        tracking_params['TransformParams']['mode'],
        tracking_params['TransformParams']['n_points'],
        tracking_params['TransformParams']['subdivisions'],
    )]

    if tracking_params['TransformParams']['GEM']:
        transform += [GEMTransform()]

    return Compose(transform)

if __name__ == '__main__':
    tracking_params = {
        'TransformParams':
            {'scales': [1, 2, 5, 7, 10],
             'mode': 'spherical', #spherical or cubical
             'n_points': 32,
             'GEM': True,
             'subdivisions': 3},
        'NetworkParams':
            {'state_dict': '', # path to model weights
             },
        'Tracking':{
            'criterion': {
                'entropy': {'value': 0.9},
                'length': {'value': 500}},
            'stepsize': 0.5,            
            }
        }
    
    img_file = r'' # path to imagefile

    # seedpoint in WORLD coordinates
    seedpoint = np.array([59.3, -35.7, -161.3]) # used Normal_10.nrrd from ASOCA

    main(img_file, seedpoint, **tracking_params)