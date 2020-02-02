import os
import glob
from skimage import io
import numpy as np

import click


@click.command()
@click.option("--base-dir", required=True, help="PROBA-V root downloaded dir.")
@click.option("--out-dir", required=True, help="Output where to save the pickles.")
@click.option("--band", required=True, help="RED|NIR band.")
def data_to_pickle(base_dir, out_dir, band):
    '''
    base_dir: specifies the root probav directory (the one downloaded from probav chalenge website)
    out_dir: specifies where to place the pickles
    band: RED or NIR band
    '''

    out_dir = out_dir.rstrip()

    train_dir = os.path.join(base_dir, 'train/'+band)
    dir_list = glob.glob(train_dir+'/imgset*')

    dir_list.sort()

    input_images_LR = np.array([[io.imread(fname) for fname in sorted(glob.glob(dir_name+'/LR*.png'))]
                                for dir_name in dir_list])

    input_images_LR.dump(os.path.join(out_dir, f'LR_dataset_{band}.npy'))

    input_images_HR = np.array(
        [io.imread(glob.glob(dir_name+'/HR.png')[0]) for dir_name in dir_list])

    input_images_HR.dump(os.path.join(out_dir, f'HR_dataset_{band}.npy'))

    mask_HR = np.array([io.imread(glob.glob(dir_name+'/SM.png')[0])
                        for dir_name in dir_list])

    mask_HR.dump(os.path.join(out_dir, f'HR_mask_{band}.npy'))

    mask_LR = np.array([[io.imread(fname,)for fname in sorted(glob.glob(dir_name+'/QM*.png'))]
                        for dir_name in dir_list])

    mask_LR.dump(os.path.join(out_dir, f'LR_mask_{band}.npy'))

    train_dir = os.path.join(base_dir, 'test', band)
    dir_list = glob.glob(train_dir+'/imgset*')
    dir_list.sort()
    test_images_LR = np.array([[io.imread(fname) for fname in sorted(glob.glob(dir_name+'/LR*.png'))]
                               for dir_name in dir_list])

    test_images_LR.dump(os.path.join(out_dir, f'LR_test_{band}.npy'))

    test_mask_LR = np.array([[io.imread(fname) for fname in sorted(glob.glob(dir_name+'/QM*.png'))]
                             for dir_name in dir_list])

    test_mask_LR.dump(os.path.join(out_dir, f'LR_mask_{band}_test.npy'))

if __name__ == '__main__':
    data_to_pickle()
