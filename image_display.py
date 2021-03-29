from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch


def show_image(path):
    img = Image.open(path)
    img_arr = np.array(img)
    plt.figure(figsize=(5, 5))
    plt.imshow(np.transpose(img_arr, (0, 1, 2)))
    print("abcd")
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    plt.show()


def my_show(path):
    img = mpimg.imread(path)
    imgplot = plt.imshow(img)
    plt.show()


# show_image("flowers/daisy/5547758_eea9edfd54_n.jpg")
testvar = 0
