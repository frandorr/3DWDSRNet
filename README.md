This repository contains the code to reproduce [Satellite Image Multi-Frame Super Resolution Using 3D Wide-Activation Neural Networks](https://www.mdpi.com/2072-4292/12/22/3812) article. 

 # Citation
If you want to use this repo please cite:
```bibtext
@article{dorr2020satellite,
  title={Satellite Image Multi-Frame Super Resolution Using 3D Wide-Activation Neural Networks},
  author={Dorr, Francisco},
  journal={Remote Sensing},
  volume={12},
  number={22},
  pages={3812},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

# 3DWDSRNet
Easy, fast and low on resources method that achieved on 02-02-2020 first place with a score of 0.94625 in [Post Mortem Proba-V Super Resolution competition](https://kelvins.esa.int/proba-v-super-resolution-post-mortem/leaderboard/).

## Update 15-03-2020
- I'm happy to announce https://github.com/mmbajo/PROBA-V scored higher (0.9416) in the leaderboard by using and improving this method. Congratulations!  

# Requirements
```python
click
tensorflow
tensorflow-addons
scikit-image
pandas
numpy
matplotlib
tqdm
```

# Usage
## Preprocessing
```sh
python dataset/dataset_to_pickle.py --base-dir probav_data \
                                    --out-dir dataset \
                                    --band NIR

python dataset/preprocessing.py --pickles-dir <pickle-dataset> \
                                --band NIR \
                                --output <patches-output>
```

## Training

The training was done in a computer with **32GB RAM** and a **GTX1050** with **4GB** of memory. If you have better specs you can adjust batch size, number of frames per images and network layers to improve the results. 

Trained models are saved on **ckpt** dir.

```sh
python train.py --band NIR \ 
                --x-patches-file dataset_NIR_X_patches.npy \
                --x-merged-patches-file dataset_NIR_X_merged_patches.npy \ 
                --y-patches-file dataset_NIR_y_patches.npy \
                --y-mask-patches-file dataset_NIR_y_mask_patches.npy \
                --checkpoint-dir ckpt/wdsr-32-8-6-7-nir-all \
                --log-dir logs/wdsr-32-8-6-7-nir-all
```

## Prediction
```sh
python predict.py --band NIR \
                  --patches-file dataset_NIR_X_test_patches.npy \ 
                  --merged-patches-file dataset_NIR_X_test_merged_patches.npy \
                  --checkpoint-dir ckpt/wdsr-32-8-6-7-nir-all \
                  --output results
```


# Introduction
As a briefly start I want to enumerate the steps I follow to reach the final solution. Most of them guided me to a deadend, but they were however necessary to achieve the final result. Explanation of architecture and params used can be found further in the next section.

| Net           | Data          | Blocks | Filters  | Loss | Normalization |Score |
| ------------- |:-------------:| -----:| -----:|-----:|-----:|-----:|
| 3DSRnet     | Full image | 3~8 |16,32,64  |MSE  | -  |~0.99  |
| 3DWDSRnet      | Full image      |   3~8 |16,32,64    |MSE    | -  |~0.99    |
| 3DWDSRnet | Full Image   | 3~8 |16,32,64  | mMSE  | Weight  |~0.98  |
| 3DWDSRnet | Patches 34x34  | 8 | 32,64  | mL1  | Weight  |~0.97  |
| 3DWDSRnet | Augmented Patches 34x34, 5 frames | 8 | 32  | mL1  | Weight  |~0.96  |
| 3DWDSRnet | Augmented Patches 34x34, 7 frames | 8 | 32  | mL1  | Weight  |~0.946  |



In the next section I'll explain only the final network.
 



# 3DWDSRnet model

The solution is based on the well known [Wide Activation for Efficient and Accurate Image Super-Resolution (1)](https://arxiv.org/abs/1808.08718) (WDSR) architecture and the framework proposed in [3DSRnet (2)](https://arxiv.org/abs/1812.09079).

The problem was treated as a Video Super Resolution problem because it has a lot of things in common:
- Many LR frames to generate one SR image
- LR frames could be shifted by one pixel  (Similar to motion in Video frames)

Taking these things into account, 3D Conv layers could take advantage of the temporal dimension and small variations in frame position. That's the idea behind 3DSRnet. But 3DSRnet uses vanilla 3DConvs as building blocks and a residual path with a Bicubic interpolation as shown in the image below.

![alt text](images/3DSRnet.png "3DSRnet framework. Image from original paper (2)")

Here I propose an improvement replacing those blocks with wdsr-b blocks.

## Replacing blocks
As stated in [(1)](https://arxiv.org/abs/1808.08718), WDSR is an architecture that can improve the EDSR performance keeping a low number of parameters to train. That's why I thought that a 3D adaptation of its building blocks could be a great fit to the 3DSRnet framework. Also, WDSR uses as a residual path the original LR frame and a Conv applied to it. I followed the same idea, and replaced the bicubic residual path with a 2DConv applied to the image's frames mean.

So, the final architecture of 3DWDSRnet looks like the image above but replacing the Bicubic Upsampling by 3DConvs and 3D-CNN Feature Extraction by WDSR-B blocks. At the end of the main and residual paths PixelShift layers were used to reconstruct the HR frame.

The proposed architecture was implemented in TensorFlow 2.0.


## Building blocks

The network is composed by two paths.
The main path is composed by one entry Convolutional 3D layer followed by `n` wdsr-b blocks with following shape:

![alt text](images/wdsr-b-block.png "wdsr-b block from (2)")

In a simplified pseudocode version, the network would be as follow:

```python
wdsr_3d(img_inputs, mean_intputs):

    # Main path
    x = Conv3D(img_inputs)
    for i in range(n): #n is the number of wdsr blocks
        x = wdsr_b_block(x)
         
    x = PixelShift(x) # upsample LR to HR
   
    # Residual path
    y = PixelShift(mean_inputs) # upsample mean_inputs by conv2D

    return y+x
```

## Normalization
It was shown in [(2)](https://arxiv.org/abs/1808.08718) that Batch Normalization doesn't work well in SR problems, that's why they replace it with Weight Normalization.

I use the same approach and applied WeightNormalization to each Conv layer.


# Dataset

The dataset is formed by two kinds of images: NIR and RED. Both of them contains LR frames, LR masks (showing dirty pixels), HR target and HR masks.


## Preprocessing

The preprocessing steps were performed as follows:

- Register all frames from each image to the corresponding first frame using skimage 
- Remove images where all of their frames had more than 15% dirty pixels
- Select K best frames (from cleanest to dirtiest) (k=7)

# Training

## Image patches
Several approaches were tried to train the network as stated in the Introduction section table. Full image training (128x128), different kind of blocks (vanilla Conv3d instead of wdsr-blocks) and number of filters (16,32,64). But none of them worked as good as the full wdsr-blocks architecture with reduced image size (34x34 to 96x96).

For each LR (128x128) image, 16 patches were taken, each one having a size of 34x34 strided by 32x32 pixels. I chose those extra pixels in patches to guarantee that no pixel was lost by means of a pixel shift.

After taking the 16 patches per image a frame shuffling was applied. Converting 16 patches into 96 (shuffling 6 times frames position for each image). Again, by doing so, performance was improved because more details could be captured by the network.

After doing so, patches where HR mask had more than 15% dirty pixels were also removed.

## NIR and RED models
First, NIR band was trained using 32 channels, 8 residual wdsr-b blocks with expansion 6.

After a plateau was reached the model was saved and used to start the RED band model training.


## Loss Function
Because of possible pixel shifts within the LR frames and the HR frames the loss function had to be reimplemented.
Based on [DeepSUM (3)](https://arxiv.org/abs/1907.06490) I rewrote the losses to take into account all possible pixel shifts and select the minimum.

Following [Loss Functions for Image Restoration](https://arxiv.org/abs/1511.08861) approach several loss functions were tried: MSE, l1, MM-SIMD and Charbonnier. The best PSNR performance was found using l1.

## Optimizer
Nadam optimizer was used with a learning rate of `5e-4`.

# Evaluation

Because the proposed network is patch based, evaluation should be done by selecting 16 34x34 patches for each test image and feeding them into the network. Then, predictions should be reconstructed by merging resulting HR patches.



