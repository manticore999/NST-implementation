import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
import os
from utils import IMAGENET_MEAN, IMAGENET_STD

class ImageProcessor:
    @staticmethod
    def load_image(image_path, size=None):
        if not os.path.exists(image_path):
            raise FileNotFoundError

        image = cv.imread(image_path)[:, :, ::-1]

        if size:
            if isinstance(size, int):
                ratio = size / image.shape[0]
                new_width = int(image.shape[1] * ratio)
                image = cv.resize(image, (new_width, size))
            else:
                image = cv.resize(image, (size[1], size[0]))

        return (image / 255.0).astype(np.float32)

    @staticmethod
    def prepare_image(image_path, size, device):
        image = ImageProcessor.load_image(image_path, size=size)
        transform_queue = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
        image = transform_queue(image).to(device).unsqueeze(0)
        return image

    @staticmethod
    def save_optimization_result(optimization_image):
        output_image = optimization_image.squeeze(axis=0).to('cpu').detach().numpy()
        output_image = np.moveaxis(output_image, 0, 2)
        final_output_image = np.copy(output_image)
        final_output_image += np.array(IMAGENET_MEAN).reshape((1, 1, 3))
        final_output_image = np.clip(final_output_image, 0, 255).astype('uint8')
        return final_output_image[:, :, ::-1]