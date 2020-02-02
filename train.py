import gc
from WDSR3D import trainer, network, loss
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
from tensorflow.keras import datasets, layers, models, Input, Model, regularizers
import tensorflow as tf
from skimage import io
from skimage import exposure
from skimage import data, img_as_float
from tqdm import tqdm
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import click


@click.command()
@click.option("--band", required=True, help="RED|NIR band.")
@click.option("--x-patches-file", required=True, help="X patches .npy file.")
@click.option("--x-merged-patches-file", required=True, help="X merged patches .npy file.")
@click.option("--y-patches-file", required=True,help="y patches .npy file.")
@click.option("--y-mask-patches-file", required=True,help="y mask patches .npy file.")
@click.option("--checkpoint-dir", required=True,help="Checkpoint dir where model is saved.")
@click.option("--log-dir", required=True,help="Tensorboard logs dir.")
@click.option("--val-split", default=0.01, help="Validation percentage")
def train(band, x_patches_file, x_merged_patches_file,
          y_patches_file, y_mask_patches_file, checkpoint_dir, log_dir, val_split):

    model = network.wdsr_3d(3, 32, 8, 6, network.res_block_b, 7)

    trainer_nir = trainer.Trainer(model,
                                  loss=loss.l1_loss,
                                  metric=loss.psnr,
                                  optimizer=tf.keras.optimizers.Nadam(
                                      learning_rate=5e-4),
                                  checkpoint_dir=checkpoint_dir,
                                  log_dir=log_dir
                                  )

    X_train_patches = np.load(
        x_patches_file, allow_pickle=True)
    X_train_merged_patches = np.load(
        x_merged_patches_file, allow_pickle=True)
    y_train_patches = np.load(
        y_patches_file, allow_pickle=True)
    y_train_mask_patches = np.load(
        y_mask_patches_file, allow_pickle=True)

    X_train, X_val, X_train_merged, X_val_merged, y_train, y_val, y_train_mask, y_val_mask = train_test_split(
        X_train_patches, X_train_merged_patches, y_train_patches, y_train_mask_patches, test_size=val_split, random_state=42)

    del X_train_patches, X_train_merged_patches, y_train_patches, y_train_mask_patches

    gc.collect()

    trainer_nir.fit([X_train, X_train_merged],
                    [y_train, y_train_mask],
                    batch_size=32,
                    validation_data=([X_val, X_val_merged], [y_val, y_val_mask]))

if __name__ == '__main__':
    train()
