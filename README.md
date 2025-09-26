# Galaxy Morphology Classification using Deep Learning

## Introduction

Automating the classification of galaxy morphologies is vital for modern astrophysics. Large-scale sky surveys generate massive datasets, making manual analysis infeasible. Deep learning enables the efficient and accurate categorization of galaxies, which in turn helps researchers explore the evolution and large-scale structure of the universe.

## Project Overview

This repository presents a deep learning pipeline for automated galaxy morphology classification, using the **Galaxy10 DECals** dataset. The project compares cutting-edge neural network architectures for image classification, focusing on their strengths, weaknesses, and suitability for astronomical data.

## Dataset

- **Galaxy10 DECals**: 17,736 labeled RGB images (256x256) across 10 galaxy morphology classes.
- **Classes**: Examples include Spiral, Elliptical, Edge-on, Merger, and others.
- **Access**: Downloaded via the [astroNN](https://astro.nn/datasets/](https://astronn.readthedocs.io/en/latest/galaxy10.html) package.

## Models Used

| Model                  | Description                                                       | Strengths                        |
|------------------------|-------------------------------------------------------------------|----------------------------------|
| ConvNeXt-Base          | Modern CNN, blends convolution and transformer design             | High accuracy, efficient local feature learning |
| EfficientNet-B4        | Scalable CNN, balances depth, width and resolution                | Accuracy/Efficiency tradeoff     |
| Swin Transformer (Tiny)| Vision Transformer with shifted window attention                  | Captures global & local patterns |

## Training Pipeline

- **Preprocessing**: Stratified train/val/test splits, normalization, augmentation (rotations, flips, color jitter).
- **Implementation**: PyTorch, torchmetrics, sklearn for evaluation.
- **Evaluation**: Accuracy, F1-score, confusion matrix, per-class analysis.

## Results Comparison

| Model                | Test Accuracy | Macro F1 | Training Time | Notable Strengths        | Limitations                  |
|----------------------|--------------|----------|--------------|--------------------------|------------------------------|
| ConvNeXt-Base        | 83.8%        | 0.819    | Medium       | Excellent feature extraction | Requires more GPU memory     |
| EfficientNet-B4      | 82.4%        | 0.804    | Fast         | Efficient, fast training     | Slightly lower accuracy      |
| Swin Transformer     | 84.1%        | 0.828    | Slow         | Best for complex patterns    | Slow, high compute demand    |

## Insights

- **Swin Transformer** outperforms others for complex or ambiguous morphologies due to its global context awareness.
- **ConvNeXt-Base** provides strong local feature learning, making it robust for common galaxy types.
- **EfficientNet-B4** offers a great balance for efficient training on large datasets.

## Recommendations

- Choose **Swin Transformer** for highest accuracy and nuanced classification.
- Use **EfficientNet-B4** or **ConvNeXt-Base** for faster training or lower compute budgets.
- Consider ensembles or stacking for even better results.

## Future Directions

- Explore larger transformers and hybrid models.
- Integrate explainability methods for scientific insights.
- Extend to other galaxy datasets (e.g., Galaxy Zoo).
- Deploy as a web API or pipeline for real-time use.
