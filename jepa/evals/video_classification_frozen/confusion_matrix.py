
#%%
### start all axis model evaluation visualizations
# Bar chart: 8 bars for axis-wise subject classification patterns
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import sklearn.metrics
import numpy as np
from sklearn.metrics import roc_auc_score

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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    plt.title(f'Confusion Matrix (Coronal Only)\nAccuracy: {accuracy:.5f}, AUC: {auc:.3f}, Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}')
    plt.tight_layout()
    plt.savefig(output_img)  # Save the plot to file
    print(f"saved fig to {output_img}")
    plt.show()



    with open(output_txt, "w") as f:
        f.write(f"Accuracy: {accuracy:.5f}, Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}, AUC: {auc:.3f}\n")

# #%%

# # Generalized pie chart: correct vs incorrect predictions (all axes combined)
# correct_total = (df['Label'] == df['Prediction']).sum()
# incorrect_total = len(df) - correct_total
# plt.figure(figsize=(6, 6))
# plt.pie([correct_total, incorrect_total], labels=['Correct', 'Incorrect'], autopct='%1.1f%%', colors=['seagreen', 'salmon'], startangle=90)
# plt.title('Overall Prediction Accuracy (All Axes Combined)')
# plt.tight_layout()
# plt.show()

# # Generalized confusion matrix (all axes combined)
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# # calculate sensitivity and specificity
# labels_sorted = sorted(df['Label'].unique())
# cm = confusion_matrix(df['Label'], df['Prediction'], labels=labels_sorted)
# TP = cm[1, 1]
# TN = cm[0, 0]
# FP = cm[0, 1]
# FN = cm[1, 0]
# sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
# specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
# accuracy = (TP + TN) / cm.sum()
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
# fig, ax = plt.subplots(figsize=(6, 6))
# disp.plot(ax=ax, colorbar=False, cmap='Blues')
# plt.title(f'Confusion Matrix (All Axes Combined)\nAccuracy: {accuracy:.3f}, Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}')
# plt.tight_layout()
# plt.show()
# #%%
# # Confusion matrix for each axis
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# fig, axs = plt.subplots(1, len(axes), figsize=(5 * len(axes), 5))
# if len(axes) == 1:
#     axs = [axs]
# for i, axis in enumerate(axes):
#     axis_df = df[df['Axis'] == axis]
#     cm = confusion_matrix(axis_df['Label'], axis_df['Prediction'], labels=sorted(df['Label'].unique()))
#     TP = cm[1, 1]
#     TN = cm[0, 0]
#     FP = cm[0, 1]
#     FN = cm[1, 0]
#     sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
#     specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
#     accuracy = (TP + TN) / cm.sum()
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(df['Label'].unique()))
#     disp.plot(ax=axs[i], colorbar=False, cmap='Blues')
#     axs[i].set_title(f"{axis_labels.get(axis, str(axis))} Axis \nAcc: {accuracy:.3f}, Sens: {sensitivity:.3f}, Spec: {specificity:.3f}")
# plt.suptitle('Confusion Matrix per Axis')
# plt.tight_layout()
# plt.show()


# #%%
# consistency_per_file.value_counts().plot(kind='pie', labels=['Inconsistent', 'Consistent'], autopct='%1.1f%%', colors=['orange', 'lightblue'])
# plt.ylabel('')
# plt.title('Prediction Consistency Across Axes')
# plt.tight_layout()
# plt.show()
# #%% 
# # concensus prediction accuracy
# # df should have 4 columns: Label, Prediction, Fname, Axis
# # print(df.head())

# # Example analysis: Calculate accuracy per axis (fix DeprecationWarning)
# accuracy_per_axis = df.groupby('Axis').apply(lambda x: (x['Label'] == x['Prediction']).mean(), include_groups=False)
# print(accuracy_per_axis)

# # Example analysis, group by Fname to see if predictions are consistent across axes
# consistency_per_file = df.groupby('Fname').apply(lambda x: x['Prediction'].nunique() == 1, include_groups=False)
# # print(consistency_per_file)

# # Calculate accuracy per subject (Fname): majority vote of predictions vs. label
# def majority_vote_accuracy(group):
#     # Use the most common prediction as the subject's prediction
#     pred = group['Prediction'].mode()[0]
#     # Use the most common label as the subject's label
#     label = group['Label'].mode()[0]
#     return int(pred == label)

# subject_accuracy = df.groupby('Fname').apply(majority_vote_accuracy, include_groups=False)
# overall_accuracy = subject_accuracy.mean()
# print(f"Subject-level accuracy (majority vote): {overall_accuracy:.3f}")
# # Build DataFrame of subject-level predictions and labels
# subject_majority = df.groupby('Fname').agg({
#     'Prediction': lambda x: x.mode()[0],
#     'Label': lambda x: x.mode()[0]
# }).reset_index()

# # Compute confusion matrix
# labels_sorted = sorted(subject_majority['Label'].unique())
# cm = confusion_matrix(subject_majority['Label'], subject_majority['Prediction'], labels=labels_sorted)
# TP = cm[1, 1]
# TN = cm[0, 0]
# FP = cm[0, 1]
# FN = cm[1, 0]
# sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
# specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
# accuracy = (TP + TN) / cm.sum()
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
# fig, ax = plt.subplots(figsize=(5, 5))
# disp.plot(ax=ax, colorbar=False, cmap='Blues')
# ax.set_title(f'Subject-level Confusion Matrix (Majority Vote)\nAccuracy: {accuracy:.3f}, Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}')
# plt.tight_layout()
# plt.show()
# ### end all axis model evaluation visualizations


# # %%

# ### start coronalonly model evaluation visualizations
# import matplotlib.pyplot as plt
# from collections import Counter
# import pandas as pd
# import sklearn.metrics
# df = pd.read_csv("/media/backup_16TB/sean/VJEPA/a6000_output/vit_base/K400/video_classification_frozen/vit_base_mriload_k40045_allmodality_coronalonly/K400andmri_mriloader_allmodality_coronal_onlyeval_results.csv")

# # %%

# # Generalized pie chart: correct vs incorrect predictions (all axes combined)
# correct = (df['Label'] == df['Prediction']).sum()
# incorrect = len(df) - correct
# plt.figure(figsize=(5, 5))
# plt.pie([correct, incorrect], labels=['Correct', 'Incorrect'], autopct='%1.1f%%', colors=['seagreen', 'salmon'], startangle=90)
# plt.title('Overall Prediction Accuracy (Coronal only trained model)')
# plt.tight_layout()
# plt.show()

# #%%
# # Generalized confusion matrix (all axes combined)
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# labels_sorted = sorted(df['Label'].unique())
# cm = confusion_matrix(df['Label'], df['Prediction'], labels=labels_sorted)
# TP = cm[1, 1]
# TN = cm[0, 0]
# FP = cm[0, 1]
# FN = cm[1, 0]
# sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
# specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
# accuracy = (TP + TN) / cm.sum()
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
# fig, ax = plt.subplots(figsize=(5, 5))
# disp.plot(ax=ax, colorbar=False, cmap='Blues')
# ax.set_title(f'Overall Confusion Matrix (Coronal only trained model)\nAccuracy: {accuracy:.3f}, Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}')
# plt.tight_layout()
# plt.show()

# # %%

# %%
