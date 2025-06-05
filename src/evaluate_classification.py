import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
import yaml

# Import model and data loading functions
from src.train_classification import PointNetClassifier, load_h5_dataset

# Set class names
class_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model(model_path):
    """Load pretrained model"""
    model = PointNetClassifier(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def get_predictions(model, data_loader):
    """Get model predictions and features"""
    all_preds = []
    all_labels = []
    all_features = []
    
    with torch.no_grad():
        for points, labels in tqdm(data_loader, desc="Evaluating"):
            points, labels = points.to(device), labels.to(device)
            
            # 获取特征向量 (在最后的分类层之前)
            features = model.encoder(points)
            
            # 获取预测
            outputs = model.classifier(features)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_features.extend(features.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_features)

def plot_confusion_matrix(true_labels, pred_labels, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save the image
    os.makedirs("results/classification", exist_ok=True)
    plt.savefig("results/classification/confusion_matrix.png", dpi=300)
    plt.close()
    print("Confusion matrix saved to results/classification/confusion_matrix.png")

def plot_tsne_visualization(features, labels, class_names):
    """Visualize feature space using t-SNE"""
    print("Computing t-SNE dimensionality reduction...")
    
    # Sample data if it's too large
    if len(features) > 1000:
        indices = np.random.choice(len(features), 1000, replace=False)
        sampled_features = features[indices]
        sampled_labels = labels[indices]
    else:
        sampled_features = features
        sampled_labels = labels
    
    # Calculate t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(sampled_features)
    
    # Visualization
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        idx = sampled_labels == i
        plt.scatter(features_tsne[idx, 0], features_tsne[idx, 1], label=class_name, alpha=0.7)
    
    plt.legend(loc='best')
    plt.title('t-SNE Feature Visualization')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    
    # Save the image
    os.makedirs("results/classification", exist_ok=True)
    plt.savefig("results/classification/tsne_visualization.png", dpi=300)
    plt.close()
    print("t-SNE visualization saved to results/classification/tsne_visualization.png")

def visualize_sample_predictions(model, test_data, test_labels, num_samples=5):
    """Visualize sample prediction results"""
    # Randomly select samples
    indices = np.random.choice(len(test_data), num_samples, replace=False)
    samples = test_data[indices]
    true_labels = test_labels[indices]
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        samples_tensor = torch.tensor(samples, dtype=torch.float32).to(device)
        outputs = model(samples_tensor)
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.cpu().numpy()
    
    # Create a large figure to display point cloud samples and prediction results
    fig = plt.figure(figsize=(15, num_samples * 3))
    
    for i in range(num_samples):
        ax = fig.add_subplot(num_samples, 1, i+1, projection='3d')
        
        # Display point cloud
        point_cloud = samples[i]
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, alpha=0.5)
        
        # Set title to display true and predicted labels
        true_class = class_names[true_labels[i]]
        pred_class = class_names[pred_labels[i]]
        
        if true_labels[i] == pred_labels[i]:
            title = f"Sample {i+1}: Correct Prediction - {pred_class}"
            ax.set_title(title, color='green')
        else:
            title = f"Sample {i+1}: Predicted {pred_class} (True: {true_class})"
            ax.set_title(title, color='red')
        
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set view angle
        ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    os.makedirs("results/classification", exist_ok=True)
    plt.savefig("results/classification/sample_predictions.png", dpi=300)
    plt.close()
    print("Sample prediction visualization saved to results/classification/sample_predictions.png")

def plot_class_accuracy(true_labels, pred_labels, class_names):
    """Plot accuracy for each class"""
    # Calculate accuracy for each class
    class_accuracies = []
    
    for i in range(len(class_names)):
        class_mask = (true_labels == i)
        if np.sum(class_mask) > 0:  # Avoid division by zero
            class_acc = np.sum((pred_labels == i) & class_mask) / np.sum(class_mask)
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0)
    
    # Plot bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, class_accuracies)
    
    # Add label for each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2%}', ha='center', va='bottom')
    
    plt.ylim(0, 1.1)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the image
    os.makedirs("results/classification", exist_ok=True)
    plt.savefig("results/classification/class_accuracy.png", dpi=300)
    plt.close()
    print("Class accuracy plot saved to results/classification/class_accuracy.png")

def main():
    # Create results directory
    os.makedirs("results/classification", exist_ok=True)
    
    # Load test data
    print("Loading test data...")
    test_data, test_labels = load_h5_dataset('datasets/modelnet10_test_1024.h5')
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=32)
    
    # Load model
    print("Loading pretrained model...")
    model_path = "models/classification_best.pth"
    model = load_model(model_path)
    
    # Get predictions and features
    print("Getting model predictions...")
    pred_labels, true_labels, features = get_predictions(model, test_loader)
    
    # Output overall accuracy
    accuracy = np.mean(pred_labels == true_labels)
    print(f"\nOverall accuracy: {accuracy:.4f}")
    
    # Output detailed classification report
    report = classification_report(true_labels, pred_labels, target_names=class_names)
    print("\nClassification report:")
    print(report)
    
    # Save classification report to file
    os.makedirs("results/classification", exist_ok=True)
    with open("results/classification/classification_report.txt", "w") as f:
        f.write(f"Overall accuracy: {accuracy:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report)
    
    # Generate confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(true_labels, pred_labels, class_names)
    
    # Plot accuracy for each class
    print("Generating class accuracy plot...")
    plot_class_accuracy(true_labels, pred_labels, class_names)
    
    # t-SNE visualization
    print("Generating t-SNE feature visualization...")
    plot_tsne_visualization(features, true_labels, class_names)
    
    # Visualize sample predictions
    print("Generating sample prediction visualization...")
    visualize_sample_predictions(model, test_data.numpy(), test_labels.numpy())
    
    print("\nAll evaluation results have been saved to the 'results/classification' folder")

if __name__ == "__main__":
    main()
