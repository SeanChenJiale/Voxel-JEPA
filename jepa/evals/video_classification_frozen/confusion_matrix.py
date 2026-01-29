
#%%
### start all axis model evaluation visualizations
# Bar chart: 8 bars for axis-wise subject classification patterns
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import sklearn.metrics
import numpy as np
from sklearn.metrics import roc_auc_score
from evals.video_classification_frozen.calculate_f1 import calculate_f1_from_confusion_matrix
def plot_confusion_matrix(csv_path, is_multiclass=False):
    # --------- USER CONFIGURATION ---------
    # csv_path = "/media/backup_16TB/sean/VJEPA_results/base/video_classification_frozen/skip1_ftpt/skip1_ftpteval_results.csv"
    output_txt = csv_path.replace("csv","txt")
    output_img = csv_path.replace("csv","png")
    # is_multiclass = False  # Set to False for binary classification, True for multiclass
    # --------------------------------------

    df = pd.read_csv(csv_path)

    def parse_logits(logit_str):
        arr = logit_str.replace('[', '').replace(']', '').split()
        return np.array([float(x) for x in arr])

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()



    labels = df['Label'].astype(int).values
    probs = []

    for logit_str in df['Probabilities']:
        logits = parse_logits(logit_str)
        prob_vec = softmax(logits)
        probs.append(prob_vec)

    probs = np.array(probs)  # shape: [num_samples, num_classes]

    if is_multiclass:
        auc = roc_auc_score(labels, probs, multi_class='ovr')
        print(f"Multiclass AUC-ROC: {auc:.4f}")
    else:
        # For binary, use probability for positive class (class 1)
        auc = roc_auc_score(labels, probs[:, 1])
        print(f"Binary AUC-ROC: {auc:.4f}")

    # # Generalized pie chart: correct vs incorrect predictions (all axes combined)
    # correct = (df['Label'] == df['Prediction']).sum()
    # incorrect = len(df) - correct
    # plt.figure(figsize=(5, 5))
    # plt.pie([correct, incorrect], labels=['Correct', 'Incorrect'], autopct='%1.1f%%', colors=['seagreen', 'salmon'], startangle=90)
    # plt.title('Overall Prediction Accuracy (Coronal Only)')
    # plt.tight_layout()
    # plt.show()

    # Generalized confusion matrix (all axes combined)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    # calculate sensitivity and specificity
    labels_sorted = sorted(df['Label'].unique())
    cm = confusion_matrix(df['Label'], df['Prediction'], labels=labels_sorted)

    if len(cm) == 2:
        TP = cm[1, 1]
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        accuracy = (TP + TN) / cm.sum()
    if len(cm) == 3:
        # for 3x3 confusion matrix, calculate macro-averaged sensitivity and specificity
        sensitivities = []
        specificities = []
        for i in range(3):
            TP = cm[i, i]
            FN = cm[i, :].sum() - TP
            FP = cm[:, i].sum() - TP
            TN = cm.sum() - (TP + FP + FN)
            sensitivities.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
            specificities.append(TN / (TN + FP) if (TN + FP) > 0 else 0)
        sensitivity = sum(sensitivities) / 3
        specificity = sum(specificities) / 3
        accuracy = (cm.diagonal().sum()) / cm.sum()

    #calculate F1 score
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    if is_multiclass:
        f1_scores = calculate_f1_from_confusion_matrix(cm)
        f1 = f1_scores['weighted_f1'] # f1 is weighted

    else:
        f1 = f1_score(df['Label'], df['Prediction'])
        precision = precision_score(df['Label'], df['Prediction'])
        print(f"Precision: {precision:.4f}")
        print(f"F1 Score: {f1:.4f}")
    # calculate precision.


    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    
    if is_multiclass:
        plt.title(f'Confusion Matrix (Coronal Only)\nAccuracy: {accuracy:.5f}, AUC: {auc:.3f}, Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}\nF1: {f1:.4f},')
    else:
        plt.title(f'Confusion Matrix (Coronal Only)\nAccuracy: {accuracy:.5f}, AUC: {auc:.3f}, Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}\nF1: {f1:.4f}, Precision: {precision:.4f}')
    plt.tight_layout()
    plt.savefig(output_img)  # Save the plot to file
    print(f"saved fig to {output_img}")



    with open(output_txt, "w") as f:
        f.write(f"Accuracy: {accuracy:.5f}, Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}, AUC: {auc:.3f}\n")
        # f.write(f"Accuracy: {accuracy:.5f}, Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}, AUC: {auc:.3f}, F1: {f1:.4f}, Precision: {precision:.4f}\n")

