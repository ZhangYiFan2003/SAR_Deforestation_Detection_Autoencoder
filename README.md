# SAR_Deforestation_Detection_Autoencoder

This project aims to detect deforestation in SAR (Synthetic Aperture Radar) images by training an autoencoder model. The autoencoder is designed to extract essential features from multi-channel SAR data to identify signs of deforestation over time.

## Current Model Architecture
![Model Structure](https://i.imgur.com/Yyhat4z.png)

## Current Trainig Effect
![Current Effect](https://i.imgur.com/AsMkAUt.png)

## Project Structure

The project includes seven main Python scripts:

1. **`train.py`**: The primary script for training or testing the autoencoder model. It handles hyperparameter optimization (using Optuna), model training, early stopping, and result visualization.
2. **`AE.py`**: Implements the core functionalities of the autoencoder network, including the encoder, decoder, training, and validation methods.
3. **`VAE.py`**: Implements the core functionalities of a Variational Autoencoder (VAE), including the encoder, decoder, training, and validation methods. By adding probabilistic modeling of the latent space, VAE captures the data's complex distribution more effectively.
4. **`architectures.py`**: Defines the architecture components of the autoencoder, including custom modules for the encoder and decoder.
5. **`datasets.py`**: Handles data loading and preprocessing. Custom dataset classes load forest images stored in `.tif` format and transform them into a format suitable for training and evaluation.
6. **`early_stopping.py`**: Implements early stopping functionality to terminate training when validation loss no longer decreases, preventing overfitting. It uses moving average smoothing to evaluate validation loss and saves the best model weights.
7. **`loss_distribution_analyse.py`**: Analyzes the distribution of pixel-level MSE losses and visualizes MSE heatmaps during the testing phase.

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

## File Details

### train:

- Entry point for the project.
- Includes command-line arguments for various hyperparameters.
- Implements core training and testing loops.

### ae:

- Defines the autoencoder (`AE_Network` and `AE`) class.
- Implements train and validation functions.

### vae:

- Defines the Variational Autoencoder (`VAE_Network` and `VAE`) class.
- Implements reparameterization, training and validation functions.

### architectures:

- Contains `Encoder`, `Decoder`, `ResidualBlock`, and `SelfAttention` classes.
- Uses feature mapping, residual blocks, and attention mechanisms for encoding and decoding.

### datasets:

- Defines `ProcessedForestDataset` and `ProcessedForestDataLoader` classes.
- Handles preprocessing and batching of `.tif` images.

### early_stopping:

- Defines the `EarlyStopping` class.
- Implements functionality to terminate training when validation loss stagnates, preventing overfitting.

### loss_distribution_analyse:

- Defines the `LossDistributionAnalysis` class.
- Computes pixel-level MSE loss during testing and generates MSE heatmaps for analysis.

## Installation and Dependencies:

The project requires Python 3 and the following key libraries:

- **PyTorch**: For neural network implementation and training.
- **Torchvision**: For image preprocessing transformations.
- **Optuna**: For hyperparameter optimization.
- **Numpy**: For numerical computations.
- **Matplotlib**: For plotting and result visualization.
- **Pandas**: For data processing and analysis.
- **Tifffile**: For handling `.tif` images.
- **TensorBoard**: For logging and visualizing training metrics.
- **Pillow**: For image processing.

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