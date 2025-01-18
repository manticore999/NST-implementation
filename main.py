import streamlit as st
import os
import cv2 as cv
from nst_app import NeuralStyleTransferApp
from utils import get_default_config
from image_processor import ImageProcessor

# Configure page to use wide mode
st.set_page_config(
    page_title="Neural Style Transfer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add caching for model loading
@st.cache_resource
def load_model():
    # Your model loading code here
    pass

def format_image_name(filename):
    # Remove file extension
    name = os.path.splitext(filename)[0]
    # Replace underscores with spaces and capitalize words
    return name.replace('_', ' ').title()

def main():
    st.title("Neural Style Transfer")

    # Directory setup
    content_dir = "data/content"
    style_dir = "data/style"
    output_dir = "data/output"

    os.makedirs(content_dir, exist_ok=True)
    os.makedirs(style_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Get lists of images with formatted names for display
    content_images = [f for f in os.listdir(content_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    style_images = [f for f in os.listdir(style_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    
    content_name_map = {format_image_name(f): f for f in content_images}
    style_name_map = {format_image_name(f): f for f in style_images}

    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Content Image")
        use_upload_content = st.checkbox("Upload your own content image")
        
        if use_upload_content:
            content_file = st.file_uploader("Upload Content Image", type=['png', 'jpg', 'jpeg'])
            if content_file:
                content_path = content_file
                st.image(content_file, use_container_width=True)
        else:
            selected_content_display = st.selectbox("Select Content Image", sorted(content_name_map.keys()))
            if selected_content_display:
                selected_content = content_name_map[selected_content_display]
                content_path = os.path.join(content_dir, selected_content)
                st.image(content_path, use_container_width=True)

    with col2:
        st.subheader("Style Image")
        use_upload_style = st.checkbox("Upload your own style image")
        
        if use_upload_style:
            style_file = st.file_uploader("Upload Style Image", type=['png', 'jpg', 'jpeg'])
            if style_file:
                style_path = style_file
                st.image(style_file, use_container_width=True)
        else:
            selected_style_display = st.selectbox("Select Style Image", sorted(style_name_map.keys()))
            if selected_style_display:
                selected_style = style_name_map[selected_style_display]
                style_path = os.path.join(style_dir, selected_style)
                st.image(style_path, use_container_width=True)

    # Configuration and processing
    with st.sidebar:
        config = get_default_config()
        st.markdown("### Parameters")
        config['height'] = st.slider("Image Height", 200, 800, config['height'])
        config['content_weight'] = st.slider("Content Weight", 1e-1, 1e1, 1e0, format="%.1e")
        config['style_weight'] = st.slider("Style Weight", 1e-4, 1e-2, 5e-3, format="%.1e")
        config['num_iterations'] = st.slider("Number of Iterations", 100, 1000, 500, 100)

    if st.button("Start Style Transfer"):
        if 'content_path' in locals() and 'style_path' in locals():
            app = NeuralStyleTransferApp()
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                result_image = st.empty()
                loss_text = st.empty()

                def update_progress(counter, total_iterations, total_loss, content_loss, style_loss, tv_loss, current_image):
                    progress = counter / total_iterations
                    progress_bar.progress(progress)
                    status_text.text(f"Iteration {counter}/{total_iterations}")
                    
                    if counter % 2 == 0:
                        result = ImageProcessor.save_optimization_result(current_image)
                        result_image.image(result, channels="BGR", use_container_width=True)

                result = app.run_style_transfer(content_path, style_path, config, progress_callback=update_progress)
                
                output_path = os.path.join(output_dir, "styled_image.jpg")
                cv.imwrite(output_path, result)
                st.image(output_path, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please select both content and style images")

if __name__ == "__main__":
    main()