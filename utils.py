IMAGENET_MEAN = [123.675, 116.28, 103.53]
IMAGENET_STD = [1, 1, 1]

def get_default_config():
    return {
        'height': 400,
        'content_weight': 1e6,
        'style_weight': 1e3,
        'tv_weight': 1.0,
        'num_iterations': 500,
        'content_layer': ['conv4_2'],
        'style_layers': ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    }