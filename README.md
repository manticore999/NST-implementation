# Neural Style Transfer with PyTorch

This project implements Neural Style Transfer using PyTorch, based on the paper ["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576) by Gatys et al. It features a user-friendly Streamlit interface to apply the style transfer to your own images.

## How It Works

### Neural Style Transfer Algorithm

The algorithm works by optimizing a content image to simultaneously match:
- The content representation of the content image
- The style representation of the style image

#### Feature Extraction
We use a pre-trained VGG19 network, extracting features from specific layers:
- Content layer: 'conv4_2' (captures structural information)
- Style layers: ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'] (capture style at different scales)

#### Loss Functions

1. **Content Loss**
python
content_loss = torch.nn.MSELoss()(current_content, target_content)

2. **Style Loss**
```python
def compute_gram_matrix(img):
    b, ch, h, w = img.size()
    features = img.view(b, ch, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / torch.tensor(features.shape[1], dtype=torch.float32)
```
Captures texture information using Gram matrices of feature maps.

3. **Total Variation Loss**
```python
tv_loss = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum() + 
          torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
```
Promotes smoothness in the generated image.

### Optimization Parameters

The balance between content and style is controlled by weights:
- Content Weight: 1e0
- Style Weight: 5e-3
- TV Weight: 1e-4
- Number of iterations: 500

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Streamlit
- OpenCV
- NumPy

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/neural-style-transfer.git
cd neural-style-transfer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run main.py
```

### Using the Application

1. **Select Images**
   - Choose from provided content and style images
   - Or upload your own images

2. **Adjust Parameters (Optional)**
   - Image Height: Controls output resolution
   - Content Weight: Adjusts content preservation
   - Style Weight: Controls style influence
   - Number of Iterations: Affects quality and processing time

3. **Start Transfer**
   - Click "Start Style Transfer"
   - Watch the progress in real-time
   - Download the final result

## Project Structure

```
neural-style-transfer/
├── main.py              # Streamlit interface
├── models.py            # VGG19 model implementation
├── image_processor.py   # Image processing utilities
├── loss.py             # Loss functions
├── nst_app.py          # Core NST application
├── utils.py            # Utility functions
└── data/               # Image directories
    ├── content/        # Content images
    ├── style/          # Style images
    └── output/         # Generated images
```

## Implementation Details

### Image Processing
- Images are preprocessed using ImageNet normalization
- Mean values: [123.675, 116.28, 103.53]
- Aspect ratio is preserved during resizing

### Optimization
- Uses L-BFGS optimizer for better convergence
- Real-time visualization of the optimization process
- Progress tracking with loss metrics

## Results

The implementation achieves high-quality style transfer with:
- Good content preservation
- Effective style transfer
- Smooth transitions
- Minimal artifacts

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original NST paper by Gatys et al.
- PyTorch team for the deep learning framework
- Streamlit team for the web interface framework