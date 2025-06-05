import matplotlib.pyplot as plt
import numpy as np
import csv

# Data
recon_train_loss = [
    0.6999, 0.2243, 0.1493, 0.1146, 0.0789, 0.0690, 0.0625, 0.0595, 0.0557, 0.0523,
    0.0476, 0.0416, 0.0376, 0.0345, 0.0330, 0.0308, 0.0302, 0.0292, 0.0280, 0.0270,
    0.0242, 0.0237, 0.0235, 0.0236, 0.0229, 0.0233, 0.0232, 0.0226, 0.0229, 0.0224,
    0.0234, 0.0223, 0.0219, 0.0223, 0.0221, 0.0217, 0.0223, 0.0223, 0.0235, 0.0221,
    0.0204, 0.0198, 0.0194, 0.0195, 0.0198, 0.0192, 0.0192, 0.0193, 0.0196, 0.0205
]
recon_test_loss = [
    0.3624, 0.1777, 0.1393, 0.0936, 0.0763, 0.0729, 0.0655, 0.0597, 0.0635, 0.0519,
    0.0455, 0.0449, 0.0383, 0.0358, 0.0355, 0.0342, 0.0346, 0.0319, 0.0306, 0.0297,
    0.0267, 0.0254, 0.0272, 0.0257, 0.0262, 0.0259, 0.0260, 0.0251, 0.0246, 0.0262,
    0.0246, 0.0247, 0.0253, 0.0260, 0.0245, 0.0238, 0.0248, 0.0285, 0.0241, 0.0255,
    0.0232, 0.0220, 0.0224, 0.0224, 0.0219, 0.0218, 0.0224, 0.0222, 0.0297, 0.0221
]

# PCN数据 (从completion_losses.txt文件中提取前50个epoch)
pcn_train_loss = [
    0.105680, 0.063661, 0.048740, 0.039127, 0.035930, 
    0.031616, 0.029326, 0.027933, 0.026890, 0.025694,
    0.024813, 0.023897, 0.023491, 0.022222, 0.021916,
    0.021195, 0.020464, 0.020758, 0.019392, 0.018997,
    0.018687, 0.018247, 0.017806, 0.017586, 0.017073,
    0.017033, 0.016755, 0.016392, 0.016366, 0.015753,
    0.015720, 0.015551, 0.015132, 0.015265, 0.014957,
    0.014732, 0.014527, 0.014321, 0.014258, 0.014384,
    0.013443, 0.013300, 0.013170, 0.013061, 0.013155,
    0.012803, 0.012828, 0.012752, 0.012716, 0.012643
]

pcn_test_loss = [
    0.091578, 0.058634, 0.045619, 0.038386, 0.034116,
    0.030552, 0.030095, 0.027331, 0.026272, 0.025851,
    0.024866, 0.023010, 0.021894, 0.022361, 0.020993,
    0.021570, 0.019916, 0.019373, 0.019399, 0.019592,
    0.018633, 0.018454, 0.018766, 0.018406, 0.017456,
    0.016864, 0.018218, 0.016180, 0.016080, 0.016273,
    0.016453, 0.015450, 0.015398, 0.015612, 0.015053,
    0.015047, 0.015923, 0.015959, 0.014136, 0.016003,
    0.014178, 0.013487, 0.013684, 0.014176, 0.012977,
    0.013307, 0.012764, 0.013184, 0.013068, 0.012920
]

cls_train_acc = [
    0.8011, 0.9128, 0.9223, 0.9469, 0.9469, 0.9506, 0.9562, 0.9617, 0.9536, 0.9679,
    0.9679, 0.9697, 0.9682, 0.9797, 0.9699, 0.9774, 0.9780, 0.9825, 0.9759, 0.9835,
    0.9810, 0.9774, 0.9802, 0.9764, 0.9867, 0.9792, 0.9835, 0.9790, 0.9792, 0.9855,
    0.9797, 0.9832, 0.9877, 0.9820, 0.9882, 0.9905, 0.9897, 0.9905, 0.9907, 0.9870,
    0.9892, 0.9905, 0.9927, 0.9932, 0.9902, 0.9870, 0.9900, 0.9872, 0.9912, 0.9942
]
cls_test_acc = [
    0.7896, 0.7784, 0.8700, 0.8621, 0.9053, 0.9174, 0.8888, 0.8932, 0.8910, 0.9086,
    0.8943, 0.8998, 0.9042, 0.8833, 0.9053, 0.8821, 0.8987, 0.9108, 0.8943, 0.9141,
    0.8998, 0.8943, 0.9097, 0.8976, 0.9119, 0.8921, 0.9097, 0.9152, 0.9097, 0.9086,
    0.9130, 0.8855, 0.8998, 0.8987, 0.8932, 0.9031, 0.9108, 0.9053, 0.8910, 0.9108,
    0.9141, 0.8932, 0.9152, 0.9141, 0.9042, 0.9152, 0.8976, 0.9108, 0.8855, 0.9130
]

# Create epochs array (starting from 0 to match example)
epochs = list(range(0, len(recon_train_loss)))

# Create figure with two y-axes
fig, ax1 = plt.subplots(figsize=(6, 6))
ax2 = ax1.twinx()

# Set style to match example
plt.rcParams['font.family'] = 'Arial'
ax1.set_facecolor('white')
ax1.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)

# Plot classification accuracy data (blue lines)
line1, = ax1.plot(epochs, cls_train_acc, color='blue', linewidth=1.5, label='Train Accuracy')
line2, = ax1.plot(epochs, cls_test_acc, color='blue', linewidth=1.5, linestyle='--', label='Test Accuracy')

# Plot reconstruction loss data (red lines)
line3, = ax2.plot(epochs, recon_train_loss, color='red', linewidth=1.5, label='FDN Train Loss')
line4, = ax2.plot(epochs, recon_test_loss, color='red', linewidth=1.5, linestyle='--', label='FDN Test Loss')

# Plot PCN loss data (green lines)
line5, = ax2.plot(epochs, pcn_train_loss, color='green', linewidth=1.5, label='PCN Train Loss')
line6, = ax2.plot(epochs, pcn_test_loss, color='green', linewidth=1.5, linestyle='--', label='PCN Test Loss')

# Set labels and title
ax1.set_xlabel('Training Epochs', fontsize=11)
ax1.set_ylabel('Classification Accuracy', color='blue', fontsize=11)
ax2.set_ylabel('Loss (Chamfer distance)', color='black', fontsize=11)
plt.title('Chamfer distance v.s. classification accuracy on ModelNet10', fontsize=13)

# Set limits and ticks
ax1.set_ylim(0.65, 1.0)
ax2.set_ylim(0, 0.2)
ax1.set_xlim(0, 50)

# Set tick colors
ax1.tick_params(axis='y', colors='blue')
ax2.tick_params(axis='y', colors='red')

# Create legend
lines = [line1, line2, line3, line4, line5, line6]
labels = ['Cls Train Accuracy', 'Cls Test Accuracy', 'FDN Train Loss', 'FDN Test Loss', 'PCN Train Loss', 'PCN Test Loss']
plt.legend(lines, labels, loc='center right', frameon=True, fontsize=8)

# Save the figure and show it
plt.tight_layout()
plt.savefig('classification_vs_reconstruction_with_pcn.png', dpi=300, bbox_inches='tight')
plt.show()