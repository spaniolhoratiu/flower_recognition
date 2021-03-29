from PIL import Image
from pytorch_training_cnn_model import cnn_model
from image_preprocessing import transformations


# test_image = Image.open("flowers/dandelion/13920113_f03e867ea7_m.jpg") works
# test_image = Image.open("flowers/dandelion/8475758_4c861ab268_m.jpg") doesnt work
# test_image = Image.open("flowers/daisy/5673728_71b8cb57eb.jpg") works
def predict(image_name):
    test_image = Image.open(image_name)
    test_image_tensor = transformations(test_image).float()
    test_image_tensor = test_image_tensor.unsqueeze_(0)
    output = cnn_model(test_image_tensor)
    class_index = output.data.numpy().argmax()
    flower_name = number_to_flower_name(class_index)
    return flower_name


def number_to_flower_name(class_index):
    switcher = {
        0: "daisy",
        1: "dandelion",
        2: "rose",
        3: "sunflower",
        4: "tulip"
    }
    return switcher.get(class_index)

