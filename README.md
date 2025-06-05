# 3D Point Cloud Classification and Reconstruction

This project focuses on applying deep learning techniques to classify and reconstruct 3D point cloud data. Based on the ModelNet10 dataset, we utilize Point#### 2. Grouping


---

## 📁 Project Structure

```
project/
├── src/                        # Source code files
│   ├── pointnet2_cls.py        # Classification model (PointNet++)
│   ├── foldingnet_ae.py        # FoldingNet + VAE decoder
│   ├── dataloader.py           # Data loading and preprocessing
│   ├── train_classification.py # Training classifier
│   ├── train_reconstruction.py # Training AE/VAE for reconstruction
│   ├── train_completion.py     # Training point cloud completion model
│   ├── evaluate_classification.py  # Evaluating classification performance
│   ├── evaluate_reconstruction.py  # Evaluating reconstruction performance
│   ├── evaluate_completion.py  # Evaluating completion performance
│   └── utils.py                # Helper functions and utilities
│
├── configs/                    # Configuration files
│   └── config.yaml             # Hyperparameters and path configs
│
├── models/                     # Saved models
│   ├── classification_best.pth # Best classification model
│   ├── autoencoder_best.pth    # Best autoencoder model
│   └── PCN_best.pth            # Best PCN model
│
├── results/                    # Results directory
│   ├── classification/         # Classification results
│   │   ├── class_accuracy.png
│   │   ├── classification_report.txt
│   │   ├── confusion_matrix.png
│   │   └── tsne_visualization.png
│   │
│   ├── reconstruction/         # Reconstruction results
│   │
│   └── visualization/          # Visualization results
│       ├── cd_loss_per_class.png
│       ├── chair_visibility_comparison.png
│       └── table_visibility_comparison.png
│
├── datasets/                   # Downloaded datasets (manually added or downloaded via code)
│
├── docs/                       # Documentation and papers
│   ├── 1712.07262v2.pdf        # PointNet++ paper
│   └── 1808.00671v3.pdf        # FoldingNet paper
│
└── environment.yml             # Conda environment definition
```

---

## 📦 Environment Requirements

This project uses:

- Python 3.8
- PyTorch (supporting Apple Silicon's MPS backend)
- Open3D
- scikit-learn
- h5py, matplotlib, tqdm, pyyaml

All dependencies are included in the `environment.yml` file.

### Installation

```bash
conda env create -f environment.yml
conda activate TIF360_proj
```

Alternatively, you can install dependencies using pip:

```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Data Preparation

```bash
python src/generate_dataset.py --output_dir datasets
```

### Train Classifier

```bash
python src/train_classification.py
```

### Train Reconstruction Model

```bash
python src/train_reconstruction.py
```

### Evaluate Models

```bash
python src/evaluate_classification.py
python src/evaluate_reconstruction.py
```

### Visualize Results

```bash
python src/visualize_cd_loss.py
```

---

## 📊 Results

- Classification accuracy: See `results/classification/classification_report.txt`
- Reconstruction performance: See results in the `results/reconstruction/` directory
- Visual examples: View images in the `results/visualization/` directory

---

## Project Overview

Understanding and reconstructing 3D shapes from data is a fundamental challenge in robotics, virtual reality, and computer vision. This project introduces deep learning techniques for 3D shape classification and reconstruction using the ModelNet10 dataset, one of the most widely used benchmarks for point cloud analysis. We develop and evaluate models for object classification and explore methods for reconstructing incomplete 3D shapes from limited or occluded data. Through this project, we gain practical experience in 3D deep learning and understand the challenges and techniques involved in recognizing and reconstructing 3D objects from partial information.

Reference: 3D ShapeNets: A Deep Representation for Volumetric Shapes, https://arxiv.org/abs/1406.5670 

---

## Project Objectives

Develop and compare deep learning models for 3D shape classification to evaluate their performance in accurately identifying objects. Additionally, design a deep learning model capable of completing missing parts of 3D objects given incomplete point cloud data. This can be achieved by deliberately removing portions of the data or by adjusting the view of objects.

---

## Dataset

The ModelNet dataset is widely available from various sources. In this study, we focus on its smaller variant, ModelNet10, which contains 10 different object categories.
- https://3dshapenets.cs.princeton.edu/
- https://modelnet.cs.princeton.edu/
- https://www.kaggle.com/datasets/balraj98/modelnet10-princeton-3d-object-dataset/data

---

## 🔍 Detailed Results

### Classification Performance

The PointNet++ based classification model achieved an overall accuracy of 91.19% on the ModelNet10 test set. Below is a summary of the per-class performance:

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| bathtub     | 0.98      | 0.86   | 0.91     | 50      |
| bed         | 0.93      | 0.99   | 0.96     | 100     |
| chair       | 0.99      | 0.99   | 0.99     | 100     |
| desk        | 0.88      | 0.78   | 0.83     | 86      |
| dresser     | 0.83      | 0.79   | 0.81     | 86      |
| monitor     | 1.00      | 0.99   | 0.99     | 100     |
| night_stand | 0.69      | 0.78   | 0.73     | 86      |
| sofa        | 0.91      | 1.00   | 0.95     | 100     |
| table       | 0.92      | 0.87   | 0.89     | 100     |
| toilet      | 1.00      | 0.99   | 0.99     | 100     |

Key observations:
- Highest performance on monitor and toilet classes (99% F1-score)
- Most challenging class was night_stand (73% F1-score)
- Confusion primarily occurred between geometrically similar classes (desk/table, dresser/night_stand)

![Confusion Matrix](/results/classification/confusion_matrix.png)
![Class Accuracy](/results/classification/class_accuracy.png)
![t-SNE Visualization](/results/classification/tsne_visualization.png)

### Point Cloud Completion Performance

Our Point Cloud Completion Network (PCN) showed impressive performance in reconstructing complete 3D shapes from partial inputs. The model achieved a steady decrease in Chamfer Distance loss through training:

- Training started with a loss of 0.106
- By epoch 10, the loss decreased to 0.026
- Fine-grained completion showed better results than coarse reconstruction

The training progression demonstrates the model's ability to learn increasingly accurate 3D shape representations.

![CD Loss Per Class](/results/reconstruction_PCN/cd_loss_per_class.png)
![All Classes Partial Completion](/results/reconstruction_PCN/all_classes_partial_50.png)

### Reconstruction Comparisons

Comparing different visibility levels and reconstruction methods:

1. **Chair Reconstruction**: Visualization shows improved detail recovery even with only 50% visibility
   ![Chair Visibility Comparison](/results/reconstruction_PCN/chair_visibility_comparison.png)

2. **Table Reconstruction**: The network successfully reconstructs complex geometrical features
   ![Table Visibility Comparison](/results/reconstruction_PCN/table_visibility_comparison.png)

3. **Comparison with FoldingNet**: PCN demonstrates superior performance in maintaining fine details
   ![PCN vs FoldingNet](/results/reconstruction_PCN/fine_pcn_vs_fdn.png)

4. **Classification vs Reconstruction**: Joint analysis showing correlation between classification confidence and reconstruction accuracy
   ![Classification vs Reconstruction](/results/reconstruction_PCN/classification_vs_reconstruction_with_pcn.png)

The results demonstrate that our proposed approach effectively handles 3D shape classification and reconstruction from partial point clouds, with particularly strong performance on objects with distinctive geometric features.