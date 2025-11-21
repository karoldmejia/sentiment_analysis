import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


# Training curves (accuracy and loss)
# Training curves (accuracy and loss)
def plot_training_history(history, title="Training History", show=True):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#F4C2C2', '#FF69B4']  # Train y Val

    # Accuracy
    ax[0].plot(history.history['accuracy'], label='Train Accuracy', color=colors[0])
    ax[0].plot(history.history['val_accuracy'], label='Val Accuracy', color=colors[1])
    ax[0].set_title('Accuracy over epochs')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    ax[0].grid(True)

    # Loss
    ax[1].plot(history.history['loss'], label='Train Loss', color=colors[0])
    ax[1].plot(history.history['val_loss'], label='Val Loss', color=colors[1])
    ax[1].set_title('Loss over epochs')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    ax[1].grid(True)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if show:
        plt.show()
    
    return fig

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
def plot_confusion_matrix(y_true, y_pred, labels=["Negative", "Positive"], title="Confusion Matrix", show=True):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=sns.color_palette(['#F4C2C2', '#FF69B4'], as_cmap=True),
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    if show:
        plt.show()
    return fig

def plot_roc_curve(y_true, y_pred_prob, title="ROC Curve", show=True):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#FF69B4', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='#F4C2C2', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True)
    if show:
        plt.show()
    return fig

def plot_precision_recall(y_true, y_pred_prob, title="Precision-Recall Curve", show=True):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color='#FF69B4', lw=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.grid(True)
    if show:
        plt.show()
    return fig

# Comparison of metrics across models
def compare_metrics(results_dict):
    metrics = ["accuracy", "precision", "recall", "f1"]
    models = list(results_dict.keys())
    
    data = {metric: [results_dict[m][metric] for m in models] for metric in metrics}
    x = np.arange(len(models))
    width = 0.2

    colors = ['#F4C2C2', '#FF69B4', '#FFC0CB', '#B2E2F7']

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, data[metric], width, label=metric.capitalize(), color=colors[i % len(colors)])

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(models)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    return fig