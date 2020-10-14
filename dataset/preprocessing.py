import pandas as pd
import numpy as np
from tqdm import tqdm

from skimage.feature import masked_register_translation
from scipy import ndimage as ndi
import os
import glob
import tensorflow as tf

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

logger = logging.getLogger('__name__')


def load_data(pickles_dir, band):
    logging.info("Loading pickle datasets...")
    imgs_LR = np.load(os.path.join(
        pickles_dir, f'LR_dataset_{band}.npy'), allow_pickle=True)
    masks_LR = np.load(os.path.join(
        pickles_dir, f'LR_mask_{band}.npy'), allow_pickle=True)
    imgs_HR = np.load(os.path.join(
        pickles_dir, f'HR_dataset_{band}.npy'), allow_pickle=True)
    masks_HR = np.load(os.path.join(
        pickles_dir, f'HR_mask_{band}.npy'), allow_pickle=True)

    imgs_LR_test = np.load(os.path.join(
        pickles_dir, f'LR_test_{band}.npy'), allow_pickle=True)
    masks_LR_test = np.load(os.path.join(
        pickles_dir, f'LR_mask_{band}_test.npy'), allow_pickle=True)

    # transform in a list of numpy
    imgs_LR = np.array([np.array(x) for x in imgs_LR])
    masks_LR = np.array([np.array(x) > 0 for x in masks_LR])

    imgs_LR_test = np.array([np.array(x) for x in imgs_LR_test])
    masks_LR_test = np.array([np.array(x) > 0 for x in masks_LR_test])

    imgs_HR = np.array([np.array(x) for x in imgs_HR])
    masks_HR = np.array([np.array(x) > 0 for x in masks_HR])

    return (imgs_LR, masks_LR, imgs_HR, masks_HR), (imgs_LR_test, masks_LR_test)



def augment_patch(X_patch,y_patch,y_mask, n):
    # take n patches shuffling k patches from one patch
    frames = np.expand_dims(X_patch,0)
    new_patches = np.expand_dims(X_patch,0)

    frames_n = frames.shape[1]
    for i in range(n):
        permutation = np.random.permutation(frames_n)
        new_patches = np.ma.concatenate((new_patches,frames[:,permutation,:,:]))
    return new_patches, np.array([y_patch]*(n+1)), np.array([y_mask]*(n+1))
    

def extract_image_patches(ksize_rows, ksize_cols, strides_rows, strides_cols,images, patches_per_image=16):
    image_shape = tf.shape(images)
    new_shape = image_shape[0]*patches_per_image
    # The size of sliding window
    ksizes = [1, ksize_rows, ksize_cols, 1] 

    # How far the centers of 2 consecutive patches are in the image
    strides = [1, strides_rows, strides_cols, 1]
    rates = [1, 1, 1, 1] # sample pixel consecutively
    
    image_patches = tf.image.extract_patches(images, ksizes, strides, rates, 'VALID')
    return tf.reshape(image_patches, [tf.cast(new_shape,tf.int32), ksize_rows, ksize_cols, image_shape[3]]).numpy()
            
        


def register_frame(frame, mask, reference):

    detected_shift = masked_register_translation(reference, frame, mask)
    shifted_frame = ndi.shift(frame, detected_shift, mode='reflect')
    shifted_mask = ndi.shift(mask, detected_shift, mode='constant', cval=0)
    return shifted_frame, shifted_mask


def register_frames_to_ref(frames, masks):
    reference = frames[np.argmax([np.count_nonzero(m) for m in masks])]
    for i, f in enumerate(frames):
        reg_f, reg_mask = register_frame(f, masks[i] > 0, reference)
        masked_f = np.ma.masked_array(reg_f, mask=~reg_mask)
        masked_f = np.ma.expand_dims(masked_f, 0)
        if i == 0:
            res_frames = masked_f
        else:
            res_frames = np.ma.concatenate((res_frames, masked_f))
    return res_frames


def check_clean_frames(frames, p=0.15):
    frames_to_remove = np.array([i for i, m in enumerate(
        frames) if np.count_nonzero(m.mask)/(128*128) > p])
    return len(frames_to_remove) < frames.shape[0]


def remove_dirty_frames(frames, p=0.15):
    frames_to_remove = np.array([i for i, m in enumerate(
        frames) if np.count_nonzero(m.mask)/(128*128) > p])
    mask = np.ones(frames.shape[0], bool)

    if np.any(frames_to_remove):
        mask[frames_to_remove] = False
        return frames[mask]  # np.delete(frames, frames_to_remove,axis=0)
    else:
        return frames


def select_k_best(frames, k=7):
    non_zero = [(i, np.count_nonzero(m)) for i, m in enumerate(frames.mask)]
    sorted_non_zero = sorted(non_zero, key=lambda x: x[1])
    best_frames = np.ma.array([frames[i[0]] for i in sorted_non_zero[:k]])
    frames_n = frames.shape[0]
    if frames_n < k:  # should repeat items because cannot complete k
        print("Not complete. Adding {} frames.".format(k - frames_n))

        for i in range(k-frames_n):
            random_i = np.random.randint(0, frames_n)
            best_frames = np.ma.concatenate(
                (best_frames, np.expand_dims(best_frames[random_i], 0)))
    return best_frames


import click


@click.command()
@click.option("--pickles-dir", required=True, help="Dir where pickles are stored.")
@click.option("--band", required=True, help="RED|NIR band.")
@click.option("--k", default=7, help="How many frames to use per image")
@click.option("--augment-n", default=6, help="Data augmentation: How many frames shuffle to perfom per image")
@click.option("--output", required=True, help="Where to store preprocessed data.")
def preprocess(pickles_dir=None, band='NIR', k=7, augment_n=6, output=None):
    train_ds, test_ds = load_data(pickles_dir, band)

    logging.info('Registering train imgs...')
    reg_X = []
    for i, img in enumerate(tqdm(train_ds[0])):
        reg_X.append(register_frames_to_ref(img, train_ds[1][i]))

    logging.info('Registering test imgs...')
    reg_X_test = []
    for i, img in enumerate(tqdm(test_ds[0])):
        reg_X_test.append(register_frames_to_ref(img, test_ds[1][i]))

    X_train = np.array(reg_X)
    y_train = train_ds[2]
    y_train_mask = train_ds[3]

    X_test = np.array(reg_X_test)

    # check for dirty images that should be eliminated (all frames corrupted)
    X_train_clean_mask = np.array(
        [check_clean_frames(frames, 0.15) for frames in X_train])
    X_test_clean_mask = np.array(
        [check_clean_frames(frames, 0.15) for frames in X_test])

    X_train = X_train[X_train_clean_mask]
    y_train = y_train[X_train_clean_mask]
    y_train_mask = y_train_mask[X_train_clean_mask]

    X_test = X_test[X_test_clean_mask]

    # select k best after removing dirty frames
    logging.info('Selecting k best frames from ds...')
    X_train = np.ma.array(
        [select_k_best(remove_dirty_frames(frames, 0.15), k) for frames in X_train])
    X_test = np.ma.array(
        [select_k_best(remove_dirty_frames(frames, 0.15), k) for frames in X_test])

    # Augment training
    logging.info('Augmenting dataset shuffling frames...')
    augment_n = 6
    ori_shape = X_train.shape[0]
    train_ds = list(zip(X_train, y_train, y_train_mask))
    res = list(zip(*[augment_patch(x, y, y_mask, augment_n)
                     for x, y, y_mask in tqdm(train_ds)]))

    # Reshape to original shape
    X_train = np.ma.array(res[0]).reshape(ori_shape*(augment_n+1),7,128,128)
    y_train = np.ma.array(res[1]).reshape(ori_shape*(augment_n+1),384,384)
    y_train_mask = np.ma.array(res[2]).reshape(ori_shape*(augment_n+1),384,384)

    # Take frames mean to use as residual path in network
    X_train_merged = np.mean(X_train.data,axis=1)
    X_test_merged =np.mean(X_test.data,axis=1)

    # Reshape to channels last, needed for the net
    X_train_reshaped = np.asarray([np.moveaxis(x[:,:,:],0,-1) for x in X_train])
    y_train_reshaped = np.asarray(y_train)
    X_test_reshaped = np.asarray([np.moveaxis(x[:,:,:],0,-1) for x in X_test])

    # Expand dims
    X_train_reshaped = np.expand_dims(X_train_reshaped,4)
    X_test_reshaped = np.expand_dims(X_test_reshaped,4)

    X_train_merged = np.expand_dims(X_train_merged ,3)  
    X_test_merged = np.expand_dims(X_test_merged ,3)

    y_train_reshaped = np.expand_dims(y_train_reshaped,3)
    y_train_mask = np.expand_dims(y_train_mask,3)

    # Convert to float
    X_train_reshaped = X_train_reshaped.astype(np.float32)
    y_train_reshaped = y_train_reshaped.astype(np.float32)
    X_test_reshaped = X_test_reshaped.astype(np.float32)

    logging.info('Saving before extract patches...')
    np.save(os.path.join(output,f'dataset_{band}_X_train_aug_full.npy'),X_train_reshaped, allow_pickle=True)
    np.save(os.path.join(output,f'dataset_{band}_X_train_merged_aug_full.npy'),X_train_merged, allow_pickle=True)
    np.save(os.path.join(output,f'dataset_{band}_y_train_aug_full.npy'),y_train_reshaped, allow_pickle=True)
    np.save(os.path.join(output,f'dataset_{band}_y_train_mask_aug_full.npy'),y_train_mask.data, allow_pickle=True)
    np.save(os.path.join(output,f'dataset_{band}_X_test_reshaped_full.npy'),X_test_reshaped, allow_pickle=True)
    np.save(os.path.join(output,f'dataset_{band}_X_test_merged_full.npy'),X_test_merged, allow_pickle=True)

    with tf.device('/cpu:0'):
        pad_images = tf.pad(X_train_reshaped.squeeze(4),[[0,0],[1,1],[1,1],[0,0]],mode='REFLECT')
        pad_merged_images = tf.pad(X_train_merged,[[0,0],[1,1],[1,1],[0,0]],mode='REFLECT')


        X_train_patches = extract_image_patches(34,34,32,32,pad_images)
        X_train_merged_patches = extract_image_patches(34,34,32,32,pad_merged_images)
        y_train_patches = extract_image_patches(96,96,96,96,y_train_reshaped)
        y_train_mask_patches = extract_image_patches(96,96,96,96,y_train_mask.astype(float))

        pad_images = tf.pad(X_test_reshaped.squeeze(4),[[0,0],[1,1],[1,1],[0,0]],mode='REFLECT')
        pad_merged_images = tf.pad(X_test_merged,[[0,0],[1,1],[1,1],[0,0]],mode='REFLECT')

        X_test_patches = extract_image_patches(34,34,32,32,pad_images)
        X_test_merged_patches = extract_image_patches(34,34,32,32,pad_merged_images)

        # remove patches when hr mask has more than 15% undefined pixels
        patches_to_remove = [i for i,m in enumerate(y_train_mask_patches) if np.count_nonzero(m)/(96*96) < 0.85]
        X_train_patches = np.delete(X_train_patches,patches_to_remove,axis=0)
        X_train_merged_patches =  np.delete(X_train_merged_patches ,patches_to_remove,axis=0)
        y_train_patches =  np.delete(y_train_patches,patches_to_remove,axis=0)
        y_train_mask_patches =  np.delete(y_train_mask_patches,patches_to_remove,axis=0)

        # Expand dims needed for input in CNN
        X_train_patches = np.expand_dims(X_train_patches,4)

        logging.info('Saving patches...')
        np.save(os.path.join(output,f'dataset_{band}_X_patches.npy'),X_train_patches, allow_pickle=True)
        np.save(os.path.join(output,f'dataset_{band}_X_merged_patches.npy'),X_train_merged_patches, allow_pickle=True)
        np.save(os.path.join(output,f'dataset_{band}_y_patches.npy'),y_train_patches, allow_pickle=True)
        np.save(os.path.join(output,f'dataset_{band}_y_mask_patches.npy'),y_train_mask_patches, allow_pickle=True)
        np.save(os.path.join(output,f'dataset_{band}_X_test_patches.npy'),X_test_patches, allow_pickle=True)
        np.save(os.path.join(output,f'dataset_{band}_X_test_merged_patches.npy'),X_test_merged_patches, allow_pickle=True)

if __name__ == '__main__':
    preprocess()


