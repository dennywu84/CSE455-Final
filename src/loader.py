import json
import numpy as np
import pandas as pd
import os
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor



raw_data = "/kaggle/input/food-recognition-2022/raw_data"

def json_to_csv(annotations_path, img_path):
    with open(annotations_path, 'r') as f:
        data = json.load(f)

    images_df = pd.DataFrame(data['images'])
    annotations_df = pd.DataFrame(data['annotations'])

    merged_df = pd.merge(images_df, annotations_df, left_on='id', right_on='image_id')

    result_df = merged_df[['file_name', 'category_id']]

    # map category names
    category_mapping = {category['id']: category['name_readable'] for category in data['categories']}
    result_df['category_name'] = result_df['category_id'].map(category_mapping)

    # update paths
    result_df['file_name'] = img_path + result_df['file_name']

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
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
