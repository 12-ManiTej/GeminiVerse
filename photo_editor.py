import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Function to convert image to grayscale
def convert_to_gray(image):
    if image is not None:
        converted_img = np.array(image.convert('RGB'))
        gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
        return gray_scale
    else:
        return None

# Function to convert image to black and white
def convert_to_black_and_white(image, threshold):
    if image is not None:
        converted_img = np.array(image.convert('RGB'))
        gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(gray_scale, threshold, 255, cv2.THRESH_BINARY)
        return blackAndWhiteImage
    else:
        return None

# Function to convert image to pencil sketch
def convert_to_pencil_sketch(image, intensity):
    if image is not None:
        converted_img = np.array(image.convert('RGB')) 
        gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
        inv_gray = 255 - gray_scale
        blur_image = cv2.GaussianBlur(inv_gray, (intensity, intensity), 0, 0)
        sketch = cv2.divide(gray_scale, 255 - blur_image, scale=256)
        return sketch
    else:
        return None

# Function to apply blur effect to the image
def apply_blur_effect(image, intensity):
    if image is not None:
        converted_img = np.array(image.convert('RGB'))
        blur_image = cv2.GaussianBlur(converted_img, (intensity, intensity), 0, 0)
        return blur_image
    else:
        return None

# Main function for Streamlit app
def main():
    st.set_page_config(
        page_title="Photo Converter App",
        page_icon="📷",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Photo Converter App")
    st.sidebar.title("Options")

    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Display 'Before' and 'After' columns
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Before")
            st.image(image, use_column_width=True)

        with col2:
            st.subheader("After")
            filter_type = st.sidebar.selectbox('Convert your photo to:', ['Original', 'Gray Image', 'Black and White', 'Pencil Sketch', 'Blur Effect'])

            if filter_type == 'Gray Image':
                gray_image = convert_to_gray(image)
                if gray_image is not None:
                    st.image(gray_image, use_column_width=True)
                else:
                    st.error("Error converting image to grayscale.")
            elif filter_type == 'Black and White':
                threshold = st.sidebar.slider('Threshold', 0, 255, 127)
                bw_image = convert_to_black_and_white(image, threshold)
                if bw_image is not None:
                    st.image(bw_image, use_column_width=True)
                else:
                    st.error("Error converting image to black and white.")
            elif filter_type == 'Pencil Sketch':
                intensity = st.sidebar.slider('Intensity', 25, 255, 125, step=2)
                sketch_image = convert_to_pencil_sketch(image, intensity)
                if sketch_image is not None:
                    st.image(sketch_image, use_column_width=True)
                else:
                    st.error("Error converting image to pencil sketch.")
            elif filter_type == 'Blur Effect':
                intensity = st.sidebar.slider('Intensity', 5, 81, 33, step=2)
                blur_image = apply_blur_effect(image, intensity)
                if blur_image is not None:
                    st.image(blur_image, use_column_width=True)
                else:
                    st.error("Error applying blur effect to the image.")
            else:
                st.image(image, use_column_width=True)

if __name__ == "__main__":
    main()
