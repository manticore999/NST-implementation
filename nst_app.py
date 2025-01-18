import torch
from torch.optim import LBFGS
from torch.autograd import Variable
import numpy as np
from models import VGG19
from image_processor import ImageProcessor
from loss import LossFunctions

class NeuralStyleTransferApp:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = VGG19().to(self.device)

    def run_style_transfer(self, content_image_path, style_image_path, config, progress_callback=None):
        content_optimizing_layer = config['content_layer']
        style_optimizing_layers = config['style_layers']

        content_image = ImageProcessor.prepare_image(content_image_path, config['height'], self.device)
        style_image = ImageProcessor.prepare_image(style_image_path, config['height'], self.device)

        gaussian_noise_image = np.random.normal(
            loc=0, scale=90.0, size=content_image.shape
        ).astype(np.float32)
        init_image = torch.from_numpy(gaussian_noise_image).float().to(self.device)
        optimizing_image = Variable(init_image, requires_grad=True)

        content_set_of_feature_maps = self.model(content_image)
        style_set_of_feature_maps = self.model(style_image)

        content_dict = {
            content_optimizing_layer[0]: content_set_of_feature_maps[content_optimizing_layer[0]].squeeze(axis=0)
        }

        style_dict = {
            layer_name: LossFunctions.compute_gram_matrix(x)
            for layer_name, x in style_set_of_feature_maps.items()
            if layer_name in style_optimizing_layers
        }

        target_representations = [content_dict, style_dict]
        optimizer = LBFGS([optimizing_image], max_iter=config['num_iterations'])

        counter = 0

        def closure():
            nonlocal counter
            current_set_of_feature_maps = self.model(optimizing_image)
            current_content_dict = {
                content_optimizing_layer[0]: current_set_of_feature_maps[content_optimizing_layer[0]].squeeze(0)
            }

            current_style_dict = {
                layer_name: LossFunctions.compute_gram_matrix(x)
                for layer_name, x in current_set_of_feature_maps.items()
                if layer_name in style_optimizing_layers
            }

            current_representations = [current_content_dict, current_style_dict]

            if torch.is_grad_enabled():
                optimizer.zero_grad()

            total_loss, content_loss, style_loss, total_variation_loss = LossFunctions.compute_total_loss(
                optimizing_image, current_representations, target_representations, style_optimizing_layers,
                content_optimizing_layer, config)

            if total_loss.requires_grad:
                total_loss.backward()

            if progress_callback:
                progress_callback(counter, config['num_iterations'], total_loss, content_loss, style_loss, total_variation_loss, optimizing_image)

            counter += 1
            return total_loss

        optimizer.step(closure)
        return ImageProcessor.save_optimization_result(optimizing_image)