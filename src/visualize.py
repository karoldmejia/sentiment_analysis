import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


# Training curves (accuracy and loss)
def plot_training_history(history, title="Training History"):
    plt.figure(figsize=(12, 5))
    
    colors = ['#F4C2C2', '#FF69B4']  # Train y Val

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color=colors[0])
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color=colors[1])
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color=colors[0])
    plt.plot(history.history['val_loss'], label='Val Loss', color=colors[1])
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

# Comparison of training histories
def compare_training_histories(histories_dict, metric='accuracy', title="Training Comparison"):
    """
    histories_dict: dict with key=model_name and value=history
    metric: 'accuracy' or 'loss'
    """
    plt.figure(figsize=(8, 5))
    colors = sns.color_palette("Set2", len(histories_dict))
    
    for i, (name, history) in enumerate(histories_dict.items()):
        plt.plot(history.history[metric], label=f'{name} Train', color=colors[i], linestyle='-')
        plt.plot(history.history[f'val_{metric}'], label=f'{name} Val', color=colors[i], linestyle='--')
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.show()

# Confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels=["Negative", "Positive"], title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=sns.color_palette(['#F4C2C2', '#FF69B4'], as_cmap=True),
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

# ROC Curve and AUC
# Curva ROC y AUC
def plot_roc_curve(y_true, y_pred_prob, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#FF69B4', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='#F4C2C2', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# Precision-Recall Curve
def plot_precision_recall(y_true, y_pred_prob, title="Precision-Recall Curve"):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='#FF69B4', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(True)
    plt.show()


# Comparison of metrics across models
def compare_metrics(results_dict):
    metrics = ["accuracy", "precision", "recall", "f1"]
    models = list(results_dict.keys())
    
    data = {metric: [results_dict[m][metric] for m in models] for metric in metrics}
    x = np.arange(len(models))
    width = 0.2

    colors = ['#F4C2C2', '#FF69B4', '#FFC0CB', '#B2E2F7']

    plt.figure(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, data[metric], width, label=metric.capitalize(), color=colors[i % len(colors)])

    plt.xticks(x + width * (len(metrics) - 1) / 2, models)
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()