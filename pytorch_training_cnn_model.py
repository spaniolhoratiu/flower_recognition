from torch.optim import Adam
import torch.nn as nn
import torch
from FlowerImageClassifierNN import FlowerClassifierCNNModel
from train_test_loader import train_dataset_loader
from image_display import testvar
import test_accuracy_cnn_model
import matplotlib.pyplot as plt
import numpy as np

# cnn_model = FlowerClassifierCNNModel()
cnn_model = torch.load("saved_model.pth")
cnn_model.eval()
optimizer = Adam(cnn_model.parameters())
loss_fn = nn.CrossEntropyLoss()


def train_and_build(n_epoches):
    mean_loss_array = []
    accuracy_array = []
    for epoch in range(n_epoches):
        cnn_model.train()
        print("Epoch {}/{}".format(epoch + 1, n_epoches))
        loss_sum = 0
        loss_nr = 0
        for i, (images, labels) in enumerate(train_dataset_loader):
            optimizer.zero_grad()
            outputs = cnn_model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss
            loss_nr += 1
            # print("Loss: {} ".format(loss.item()))

        mean_loss = loss_sum / loss_nr
        mean_loss_array.append(mean_loss)
        accuracy_array.append(test_accuracy_cnn_model.test_accuracy())

    torch.save(cnn_model, "saved_model.pth")

    np_loss_array = np.array(mean_loss_array)
    np_accuracy_array = np.array(accuracy_array)
    plot_epochs = np.arange(1, n_epoches + 1)

    plt.plot(plot_epochs, np_loss_array, 'b', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(plot_epochs, np_accuracy_array, 'b', label='Accuracy')
    plt.title('Accuracy gain')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# train_and_build(200) 30-45 mins
# train_and_build(400) <1h10mins
