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

import constants

category_id_to_index = {}

def json_to_csv(annotations_path, train):
    with open(annotations_path, 'r') as f:
        data = json.load(f)

    images_df = pd.DataFrame(data['images'])
    annotations_df = pd.DataFrame(data['annotations'])

    merged_df = pd.merge(images_df, annotations_df, left_on='id', right_on='image_id')

    trim_df = merged_df[['file_name', 'category_id']]

    # filter out images with more than one label
    cat_per_image = trim_df.groupby('file_name')['category_id'].nunique()
    single_cat_images = cat_per_image[cat_per_image == 1].index

    filtered_df = trim_df[trim_df['file_name'].isin(single_cat_images)][['file_name', 'category_id']]

    if train:
        # count how many times each id shows up
        category_counts = filtered_df['category_id'].value_counts()

        # keep rows with at least MIN_CATEGORY_SIZE 
        filtered_df = filtered_df[filtered_df['category_id'].isin(category_counts[category_counts >= constants.MIN_CATEGORY_SIZE].index)]

        # map category ids to category indices
        # builds the dictionary
        unique = filtered_df['category_id'].unique() # get array of unique category ids
        for index, category_id in enumerate(unique):
            category_id_to_index[category_id] = index
    else: 
        # for validation data, filter out anything not in the dictionary
        filtered_df = filtered_df[filtered_df['category_id'].isin(category_id_to_index)]

    # uses the dictionary
    filtered_df['category_index'] = filtered_df['category_id'].map(category_id_to_index)

    # make sure its ints
    filtered_df['category_index'] = filtered_df['category_index'].astype(int)

    filtered_df = filtered_df.groupby('category_id').head(constants.MIN_CATEGORY_SIZE)

    # map category names
    category_mapping = {category['id']: category['name_readable'] for category in data['categories']}
    filtered_df['category_name'] = filtered_df['category_id'].map(category_mapping)

    return_df = filtered_df[['file_name', 'category_index', 'category_name']]

    return_df.sort_values(by='category_index', inplace=True)

    # get first couple
    answer = return_df[return_df['category_index'].between(constants.MIN_INDEX, constants.MAX_INDEX - 1)]
    print(answer)
    return answer


class CustomImageDataset(Dataset):
    def __init__(self, train, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = json_to_csv(annotations_file, train)
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
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=5, shear=[-10,10,-10,10]),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

valid_transformer = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def get_data_loader(batch_size):
    train_dataset = CustomImageDataset(True, '../data/raw_data/public_training_set_release_2.0/annotations.json', '../data/raw_data/public_training_set_release_2.0/images', transform=train_transformer)
    valid_dataset = CustomImageDataset(False, '../data/raw_data/public_validation_set_2.0/annotations.json', '../data/raw_data/public_validation_set_2.0/images', transform=valid_transformer)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader