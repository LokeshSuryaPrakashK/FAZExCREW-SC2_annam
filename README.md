# Soil Classification Script

## Overview

This project implements a *deep learning* solution for the **Soil Classification Challenge**, focusing on **binary classification** of images to determine if they depict *soil* or *not*. The script uses a pre-trained **EfficientNetB0** model with a custom classifier head, trained on a dataset of soil images. It handles multiple image formats (*jpg*, *jpeg*, *png*, *webp*, *gif*) and missing files, optimizing for the *F1-score*. The script uses **Markdown** formatting with special characters like `**` for **bold**, `*` for *italics*, and ``` for code blocks to ensure clarity on **GitHub**.

## Features

- **Data Preprocessing**: Loads images with `*augmentation*` (flips, rotations, color jitter) for training and `*normalization*` for validation/testing, handling missing files with dummy images.
- **Model**: Uses **EfficientNetB0** with `*transfer learning*`, freezing early layers and adding a custom classifier for *binary output* (*soil* vs. *non-soil*).
- **File Handling**: Supports multiple image formats and logs missing files to `*missing_files.log*`.
- **Training**: Trains with **AdamW** optimizer, `*cosine annealing scheduler*`, and *cross-entropy loss* for *2 epochs*.
- **Evaluation**: Computes *macro F1-score*, generates a `*confusion matrix*` (*confusion_matrix.png*), and plots `*training history*` (*training_history.png*).
- **Prediction**: Saves *binary predictions* for the test set in `*prakash_submission.csv*`.

## Requirements

The script requires the following, with `*` marking dependencies:

- *Python* 3.8+
- *torch*
- *torchvision*
- *pandas*
- *numpy*
- *matplotlib*
- *seaborn*
- *pillow*
- *scikit-learn*
- *tqdm*

Install using **pip**:

```bash
pip install *torch* *torchvision* *pandas* *numpy* *matplotlib* *seaborn* *pillow* *scikit-learn* *tqdm*
```

## Installation

1. **Clone the repository** with `*git clone*`:

```bash
git clone https://github.com/LokeshSuryaPrakashK/FAZExCREW-SC2_annam
cd *soil-classification-script*
```

2. **Install dependencies** with `*pip*`:

```bash
pip install -r *requirements.txt*
```

3. **Prepare the dataset** with this structure, using `#` for comments:

```
*DataSet/Soil Classification/*
├── *train/*                  # Training images
├── *test/*                   # Test images
├── *train_labels.csv*        # Image IDs and labels (*0* for *non-soil*, *1* for *soil*)
└── *test_ids.csv*            # Test image IDs
├── *prakash_sub/*            # Output directory for submission
```

Update `*Config*` in `*soil_classifier.py*` if paths differ.

## Usage

Run the script for **binary classification**:

```python
# *soil_classifier.py*
import *os*
from *soil_classifier* import *main*

# Run the pipeline with default transforms
*model*, *submission* = *main*(transform_type='*default*')

# Output: Loads training/test data, trains for *2 epochs*,
# saves `*prakash_submission.csv*`, `*best_model.pth*`, `*missing_files.log*`,
# `*confusion_matrix.png*`, and `*training_history.png*.
```

The script:
- Filters missing files, logging to `*missing_files.log*`.
- Trains **EfficientNetB0** with *F1-score* optimization.
- Generates *binary predictions* (*soil* or *non-soil*).

## Output Files

The script generates, with `**` for emphasis:
- **prakash_submission.csv**: *Binary predictions* for test set.
- **best_model.pth**: Best model weights based on *F1-score*.
- **missing_files.log**: List of missing image files.
- **confusion_matrix.png**: *Confusion matrix* visualization.
- **training_history.png**: Plots of *loss* and *F1-score*.

## Limitations

- **Single Class Issue**: If only one class is present (e.g., *Class 1*), class balancing is disabled, limiting generalization (code warns: "Only one class found").
- **Missing Files**: Handled with dummy images, which may affect performance.
- **Short Training**: Only *2 epochs*, potentially insufficient for convergence.
