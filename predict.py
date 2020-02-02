import tensorflow as tf
from WDSR3D.network import wdsr_3d, res_block_b
import numpy as np
from tqdm import tqdm
from skimage import io
import click
import os

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

logger = logging.getLogger('__name__')


def resolve(model, lr_batch, lr_mean_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    lr_mean_batch = tf.cast(lr_mean_batch, tf.float32)
    
    sr_batch = model([lr_batch,lr_mean_batch])
    sr_batch = tf.clip_by_value(sr_batch, 0, 2**16)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.float32)
    return sr_batch


def evaluate(model, X_test_patches, X_test_merged_patches):
    y_preds = []
    
    for i in tqdm(range(0, X_test_patches.shape[0], 16)):
        # Resolve
        res_patches = resolve(model,np.expand_dims(X_test_patches[i:i+16],4),X_test_merged_patches[i:i+16])
        y_pred = reconstruct_from_patches(np.array(res_patches))
        y_preds.append(y_pred)
    return y_preds


def reconstruct_from_patches(images):
    rec_img = np.zeros((384, 384, 1))
    block_n = 0
    first_block = images[0, :, :, ]
    for i in range(1, 5):
        for j in range(1, 5):

            rec_img[(i-1)*96:i*96, (j-1)*96:j*96] = images[block_n, :, :, ]
            block_n += 1

    return rec_img.reshape((384, 384, 1))

@click.command()
@click.option("--band", help="RED|NIR band.")
@click.option("--patches-file", help="Patches .npy file.")
@click.option("--merged-patches-file", help="Merged patches .npy file.")
@click.option("--checkpoint-dir", help="Checkpoint dir where model is saved.")
@click.option("--output", help="Output to save predicted images.")
def predict(band, patches_file, merged_patches_file,checkpoint_dir, output):
    try:
        (band == 'RED') | (band == 'NIR')
    except:
        raise ValueError("Band should be RED or NIR")

    k = 7
    logging.info('Loading test patches...')
    X_test_patches = np.load(patches_file, allow_pickle=True)
    X_test_merged_patches = np.load(merged_patches_file, allow_pickle=True)

    logging.info('Loading model from ckpt...')
    model = wdsr_3d(3, 32, 8, 6, res_block_b, 7)

    checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                    psnr=tf.Variable(1.0),
                                    model=model)

    checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                    directory=checkpoint_dir,
                                                    max_to_keep=5)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)

    logging.info('Evaluating...')
    y_preds = evaluate(model, X_test_patches, X_test_merged_patches)

    band = band.upper()
    if band == 'NIR':
        i = 1306
    elif band=='RED':
        i = 1160
    
    logging.info(f'Saving predicted images to {output}')
    for img in tqdm(y_preds):
        io.imsave(os.path.join(output,f'imgset{i}.png'),img[:,:,0].astype(np.uint16))
        i+=1

if __name__ == '__main__':
    predict()