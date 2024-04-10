from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import torch
import config
import utils
import cv2

class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        # print(self.list_files)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]

        # Assuming input_image and target_image are NumPy arrays
        input_image = to_pil_image(input_image)
        target_image = to_pil_image(target_image)

        # augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = config.both_transform(input_image)
        target_image = config.both_transform(target_image)

        input_image = config.transform_only_input(input_image)
        target_image = config.transform_only_input(target_image)

        return input_image, target_image

def show_img(input_img, target_img):
    display_list = [input_img, target_img]
    title = ['Input Image', 'Ground Truth']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        img_np = (display_list[i] * 0.5 + 0.5).numpy()

        plt.imshow(np.transpose(img_np, (1, 2, 0)))
        plt.axis('off')
    plt.show()

def imshow(img):
    img = img * 0.5 + 0.5  # 還原圖像
    npimg = img.numpy()

    # 顏色換至最後一維
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

def test():
    train_dataset = MapDataset(root_dir=config.TRAIN_DIR)
    ing, tar = train_dataset.__getitem__(1)

    show_img(ing, tar)
    imshow(ing)

if __name__ == "__main__":
    test()