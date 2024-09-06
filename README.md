# Dog vs Cat Image Classifier

## Overview

This project implements a deep learning model to classify images of dogs and cats using PyTorch. It utilizes transfer learning techniques with various pre-trained models to achieve high accuracy in distinguishing between dog and cat images.

## Project Structure

- `main.py`: Main script to run the training and evaluation pipeline
- `config.py`: Configuration file containing model and training parameters
- `model.py`: Definition of the neural network model and model selection
- `dataset.py`: Data loading, preprocessing, and augmentation utilities
- `train.py`: Training loop implementation with early stopping
- `evaluate.py`: Evaluation metrics calculation and misclassification display functions
- `Dog_Cat_classifier.py`: Core classifier implementation
- `requirements.txt`: List of Python dependencies
- `resnet_output_95%.png`: Sample output showing model performance

## Setup

1. Clone this repository:

2. Install the required packages:
  pip install -r requirements.txt

4. Prepare your dataset:
- Organize your images into the following structure:
  ```
  Dataset/
  ├── train/
  │   ├── dogs/
  │   └── cats/
  └── test/
      ├── dogs/
      └── cats/
  ```
- Update the `DATASET_PATH` in `config.py` if necessary.

## Usage

Run the main script with optional arguments:
  python main.py [--model MODEL] [--learning_rate LR] [--batch_size BS] [--epochs E] [--pretrained] [--freeze_base]

Arguments:
- `--model`: Choose from 'resnet18', 'resnet50', 'densenet121', 'efficientnet_b0', 'vit_base_patch16_224' (default: resnet18)
- `--learning_rate`: Set the learning rate (default: 0.001)
- `--batch_size`: Set the batch size (default: 32)
- `--epochs`: Set the number of training epochs (default: 20)
- `--pretrained`: Use pre-trained weights (flag)
- `--freeze_base`: Freeze the base model layers (flag)

Example:
  python main.py --model efficientnet_b0 --learning_rate 0.0001 --batch_size 64 --epochs 30 --pretrained --freeze_base

## Features

- Support for multiple state-of-the-art architectures
- Transfer learning with pre-trained models
- Data augmentation for improved generalization
- Early stopping to prevent overfitting
- Visualization of misclassified images
- Flexible configuration through command-line arguments

## Results

The model achieves up to 95% accuracy on the test set, as demonstrated in the `resnet_output_95%.png` file. This high accuracy is achieved through careful model selection, hyperparameter tuning, and effective use of transfer learning techniques.
![resnet_output_95%](https://github.com/user-attachments/assets/79fae3a7-eb9c-4f75-8255-60d641b56958)

## Future Improvements

- Implement cross-validation for more robust evaluation
- Add support for multi-GPU training
- Explore ensemble methods for potentially higher accuracy
- Implement a web interface for easy image upload and classification

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [torchvision](https://pytorch.org/vision/stable/index.html) for pre-trained models and utilities
- [Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats) from Kaggle for training data

## Contact
Võ Thành Nguyễn - thcs2nguyen@gmail.com

Project Link: 
