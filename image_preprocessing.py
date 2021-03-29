import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from display_transformed_image_grid import show_transformed_image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

transformations = transforms.Compose([
    transforms.RandomResizedCrop(64),  # model should work on more general pictures
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

total_dataset = datasets.ImageFolder("flowers", transform=transformations)
dataset_loader = DataLoader(dataset=total_dataset, batch_size=1000)
items = iter(dataset_loader)
image, label = items.next()

#show_transformed_image(make_grid(image))