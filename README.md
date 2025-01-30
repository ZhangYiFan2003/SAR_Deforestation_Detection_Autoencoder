# SAR_Deforestation_Detection_Autoencoder

This project aims to detect deforestation in SAR (Synthetic Aperture Radar) images by training an autoencoder model. The autoencoder is designed to extract essential features from multi-channel SAR data to identify signs of deforestation over time.

## Current Model Architecture
![Model Structure](https://i.imgur.com/Bt1axJU.png)

## Current Trainig Effect
![Current Effect](https://i.imgur.com/AsMkAUt.png)

## Current Detection result
![Current Effect](https://i.imgur.com/xA5NGhi.png)

## Dataset Structure

The project uses a dataset comprising training, validation, and test sets. All data are SAR images captured by the Sentinel-1 satellite near the Amazon region, with each pixel representing 10 meters.

- **`Training Set`**: Includes three areas, approximately 250km × 250km each, with images captured annually between **May 1 and October 1 from 2018 to 2024**. After preprocessing, all images are resized to 2 × 256 × 256 pixels, totaling about **32,000** images. The training set contains only SAR images fully covered by forests.
- **`Validation Set`**: Includes a region adjacent to the training areas, also approximately 250km × 250km. Images are captured annually between **May 1 and October 1 from 2018 to 2024**, resized to 2 × 256 × 256 pixels, with about **8,000** images in total. Like the training set, the validation set only includes SAR images fully covered by forests.
- **`Test Set`**: Includes a region adjacent to the validation area, approximately 250km × 250km. Images are captured between **May 1 and October 1 from 2020 to 2022**, resized to 2 × 256 × 256 pixels, totaling about **1,600** images. The test set includes SAR images with deforested areas.

## Workflow

### 1. Model Training and Testing

The `train.py` script is the entry point for the entire training and testing process.

- **Parameters**: `train.py` accepts multiple command-line arguments to configure model parameters such as batch size, number of training epochs, model selection (`AE` or `VAE`), learning rate, etc., allowing flexibility for experiments.
    
    Example usage:
    
    ```bash
    python train.py --train --epochs 10 --batch-size 8 --model AE --dataset FOREST --lr 0.0001
    ```
    
- **Optuna Optimization**: Use the `use-optuna` flag to apply Optuna for hyperparameter optimization to minimize validation loss. Without this flag, the training follows the specified parameters.
- **Model Training**: The `train` flag loads the specified model (`AE` or `VAE`) and trains it using the corresponding architecture from `architectures.py`.
- **Model Testing**: The `test` flag loads saved model weights for evaluation.

### 2. Autoencoder Implementation

The `AE.py` and `VAE.py` files implement the autoencoder models (AE and VAE), defining encoder and decoder classes.

- **Encoder**: Compresses input SAR data into a low-dimensional latent space using convolutional layers, residual connections, and self-attention modules.
- **Decoder**: Reconstructs input data from the latent space using upsampling blocks and self-attention modules for accurate reconstruction.
- **Differences between AE and VAE**: AE uses a standard autoencoder architecture, while VAE incorporates probabilistic modeling of the latent space, defining mean and variance to represent the distribution of latent variables. Through reparameterization, VAE samples latent variables to produce more diverse reconstructions.

### 3. Architectural Design

The encoder and decoder are defined in `architectures.py`.

- **Residual Blocks**: Use skip connections to enhance gradient flow, supporting deeper networks. These connections help prevent gradient vanishing or explosion, ensuring efficient information transfer.
- **Self-Attention Modules**: Assign attention weights across spatial dimensions to capture global relationships, aiding in detecting significant changes indicating deforestation.
- **Feature Pyramid Networks (FPN)**: Extract features at multiple scales, combining information at different resolutions to capture fine details and broader contextual features for improved reconstruction.

### 4. Dataset Loading

In `datasets.py`, custom datasets load preprocessed SAR data from `.tif` images.

- **ProcessedForestDataset Class**: Loads SAR data from a specified root directory. The dataset expects input as `.tif` images with two channels, corresponding to multi-channel SAR input.
- **DataLoader Wrapper Class**: `ProcessedForestDataLoader` wraps PyTorch DataLoader for efficient batch processing of large datasets during training, validation, and testing.

### 5. Loss and Optimization

- **AE Loss Function**: Uses Mean Squared Error (MSE) for reconstruction loss, comparing reconstructed outputs with original inputs to evaluate the model's learning capability.
- **VAE Loss Function**: Combines reconstruction loss (MSE) with Kullback-Leibler Divergence (KLD) to measure the discrepancy between the latent space and a standard normal distribution. The KLD ensures that VAE learns a smooth and reasonable latent space distribution, generating more diverse samples.

### Installing Required Libraries

Run the following command to install all dependencies:

```
pip install -r requirements.txt
```

## How to Run the Project

```
git clone <repo-url>
cd <repo-directory>
python train.py --train --epochs 20 --batch-size 16
```

## Future Improvements