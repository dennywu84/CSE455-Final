import json
import numpy as np
import pandas as pd
import os
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms


def json_to_csv(annotations_path):
    with open(annotations_path, 'r') as f:
        data = json.load(f)

    images_df = pd.DataFrame(data['images'])
    annotations_df = pd.DataFrame(data['annotations'])

    merged_df = pd.merge(images_df, annotations_df, left_on='id', right_on='image_id')

    result_df = merged_df[['file_name', 'category_id']]

    # map category names
    category_mapping = {category['id']: category['name_readable'] for category in data['categories']}
    result_df['category_name'] = result_df['category_id'].map(category_mapping)

    # result_df[['image_path', 'category_id', 'category_name']].to_csv('output.csv', index=False)

    return result_df


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = json_to_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = transforms.functional.to_pil_image(image)
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


train_transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=5, shear=[-10,10,-10,10]),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

valid_transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def get_data_loader(batch_size):
    train_dataset = CustomImageDataset('../data/raw_data/public_training_set_release_2.0/annotations.json', '../data/raw_data/public_training_set_release_2.0/images', transform=train_transformer)
    valid_dataset = CustomImageDataset('../data/raw_data/public_validation_set_2.0/annotations.json', '../data/raw_data/public_validation_set_2.0/images', transform=valid_transformer)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader