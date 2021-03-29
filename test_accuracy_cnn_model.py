import torch
from pytorch_training_cnn_model import cnn_model
from train_test_loader import test_dataset_loader
from train_test_loader import test_dataset
import sklearn.metrics
import plot_helper
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay


def test_accuracy():
    cnn_model.eval()
    test_acc_count = 0
    predictions_aux = []
    actual_labels_aux = []
    for k, (test_images, test_labels) in enumerate(test_dataset_loader):
        test_outputs = cnn_model(test_images)
        _, prediction = torch.max(test_outputs.data, 1)
        predictions_aux.append(prediction.tolist())
        actual_labels_aux.append(test_labels.data.tolist())
        test_acc_count += torch.sum(prediction == test_labels.data).item()

    test_accuracy = test_acc_count / len(test_dataset)
    print("Accuracy: {}".format(test_accuracy))

    predictions = []
    for sublist in predictions_aux:
        for item in sublist:
            predictions.append(item)
    actual_labels = []
    for sublist in actual_labels_aux:
        for item in sublist:
            actual_labels.append(item)

    print("Predictions: ", predictions)
    print("Actual labels : ", actual_labels)
    confusion_mat = sklearn.metrics.multilabel_confusion_matrix(actual_labels, predictions, labels=[0, 1, 2, 3, 4])
    print("Confusion matrix:")
    print(confusion_mat)
    plot_confusion_matrices(confusion_mat)
    return test_accuracy


def plot_confusion_matrices(confusion_mat):
    for i in range(5):
        if i == 0:
            disp_labels = ['daisy', 'not daisy']
        elif i == 1:
            disp_labels = ['dandelion', 'not dandelion']
        elif i == 2:
            disp_labels = ['rose', 'not rose']
        elif i == 3:
            disp_labels = ['sunflower', 'not sunflower']
        else:
            disp_labels = ['tulip', 'not tulip']

        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat[i], display_labels=disp_labels)
        disp = disp.plot(include_values=True,
                         cmap='viridis',
                         ax=None,
                         xticks_rotation='horizontal')
        # sklearn.metrics.plot_confusion_matrix(cnn_model, actual_labels, predictions)
        plt.show()

# test_accuracy = test_acc_count / len(test_dataset)
# print(test_accuracy)
