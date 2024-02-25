import numpy as np
import matplotlib.pyplot as plt

'''
TPR = np.load('TPR.npy')
FPR = np.load('FPR.npy')


plt.figure(figsize=(8, 6))  # Optional: You can adjust the figure size
plt.plot(FPR, TPR, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)  # Optional: Adds a grid for easier visualization
plt.show()
'''



# Assuming TPR and FPR are loaded from files
TPR = np.load('TPR.npy')
FPR = np.load('FPR.npy')

plt.figure(figsize=(8, 6))

# ROC Curve
plt.plot(FPR, TPR, color='darkorange', lw=2, label='ROC curve')

# Add points for TPR and FPR on the ROC curve
# Note: This step assumes you want to visually mark each TPR and FPR point on the ROC curve
# This might not be directly meaningful without knowing the corresponding thresholds
plt.scatter(FPR, TPR, color='red', label='TPR and FPR points')

# Diagonal line
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
