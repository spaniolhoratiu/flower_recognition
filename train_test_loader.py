from torch.utils.data import random_split
from image_preprocessing import total_dataset
from torch.utils.data import DataLoader

# Split dataset: 80% for training, 20% for testing
train_size = int(0.8 * len(total_dataset))
test_size = len(total_dataset) - train_size
train_dataset, test_dataset = random_split(total_dataset, [train_size, test_size])

# Load datasets
train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=500)
test_dataset_loader = DataLoader(dataset=test_dataset, batch_size=400)
