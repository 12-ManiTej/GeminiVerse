import streamlit as st
import os
from dotenv import load_dotenv,find_dotenv
from streamlit_option_menu import option_menu
from PIL import Image
import cv2
import numpy as np
import google.generativeai as genai
import io
import requests
from datetime import datetime
from skimage import color, exposure, restoration
load_dotenv(find_dotenv())
# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to convert image to grayscale
def gemini_pro():
    model = genai.GenerativeModel('gemini-pro')
    return model

# Load gemini vision model
def gemini_vision():
    model = genai.GenerativeModel('gemini-pro-vision')
    return model

# get response from gemini pro vision model
def gemini_visoin_response(model, prompt, image):
    response = model.generate_content([prompt, image])
    return response.text

def convert_to_gray(image):
    if image is not None:
        converted_img = np.array(image.convert('RGB'))
        gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
        return Image.fromarray(gray_scale)
    else:
        return None

# Function to convert image to black and white
def convert_to_black_and_white(image, threshold):
    if image is not None:
        converted_img = np.array(image.convert('RGB'))
        gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(
            gray_scale, threshold, 255, cv2.THRESH_BINARY)
        return Image.fromarray(blackAndWhiteImage)
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
        return Image.fromarray(sketch)
    else:
        return None

# Function to apply blur effect to the image
def apply_blur_effect(image, intensity):
    if image is not None:
        converted_img = np.array(image.convert('RGB'))
        blur_image = cv2.GaussianBlur(
            converted_img, (intensity, intensity), 0, 0)
        return Image.fromarray(blur_image)
    else:
        return None

def roleForStreamlit(user_role):
    if user_role == 'model':
        return 'assistant'
    else:
        return user_role
    
def generate_image(text):
    gemini_api_endpoint = "GOOGLE_API_KEY"  # Replace with Gemini API endpoint
    gemini_api_key = "GOOGLE_API_KEY"  # Replace with your Gemini API key
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {gemini_api_key}"
    }
    payload = {
        "text": text
    }
    response = requests.post(gemini_api_endpoint, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["url"]
    else:
        return None

def text2image(prompt):
  API_URL=(
        "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    )
  headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
  payload={
      "inputs":prompt,
  }
  response=requests.post(API_URL,headers=headers,json=payload)
  image_bytes=response.content

  image=Image.open(io.BytesIO(image_bytes))

  timestamp=datetime.now().strftime("%Y%m%d%H%M%S")
  filename=f"{timestamp}.jpg"

  image.save(filename)
  return filename

## Function to load Google Gemini Pro Vision API And get response

def get_gemini_repsonse(input,image,prompt):
    model=genai.GenerativeModel('gemini-pro-vision')
    response=model.generate_content([input,image[0],prompt])
    return response.text

def input_image_setup(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
    
def dehaze_image(image, t_min=0.1, w=0.95):
    I = image.astype('float64') / 255.0
    
    # Estimate transmission
    dark_channel = np.min(I, axis=2)
    A = np.percentile(dark_channel, 100 * (1 - t_min))
    t_est = 1 - w * dark_channel / A

    # Refine transmission
    kernel_size = 15
    t_refined = cv2.blur(t_est, (kernel_size, kernel_size))
    
    # Remove haze
    J = np.zeros_like(I)
    for i in range(3):
        J[:, :, i] = (I[:, :, i] - A) / np.maximum(t_refined, 0.1) + A

    J = np.clip(J, 0, 1)
    return (J * 255).astype('uint8')

# Function to apply dehazing effect to the image
def apply_dehazing_effect(image):
    if image is not None:
        # Convert the uploaded image data to a PIL Image object
        pil_image = Image.open(image)
        # Convert the PIL Image to a numpy array
        converted_img = np.array(pil_image)
        # Apply dehazing effect
        dehazed_img = dehaze_image(converted_img)
        # Convert the dehazed numpy array back to a PIL Image
        dehazed_pil_image = Image.fromarray((dehazed_img * 255).astype(np.uint8))
        return dehazed_pil_image
    else:
        return None
    

# Main function for Streamlit app
def main():
    st.set_page_config(
        page_title="Chat With Gemi",
        page_icon="🧠",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    with st.sidebar:
        user_picked = option_menu(
            "Google Gemini AI",
            ["ChatBot", "Image Captioning", "Image editing","Generate Image","Health App","Dehaze Effect"],
            menu_icon="robot",
            icons=["chat-dots-fill", "image-fill", "camera-reels-fill","magic"],
            default_index=0
        )

    if user_picked == 'ChatBot':
        model = gemini_pro()

        if "chat_history" not in st.session_state:
            st.session_state['chat_history'] = model.start_chat(history=[])

        st.title("🤖TalkBot")

        # Display the chat history
        for message in st.session_state.chat_history.history:
            with st.chat_message(roleForStreamlit(message.role)):
                st.markdown(message.parts[0].text)

        # Get user input
        user_input = st.chat_input("Message TalkBot:")
        if user_input:
            st.chat_message("user").markdown(user_input)
            reponse = st.session_state.chat_history.send_message(user_input)
            with st.chat_message("assistant"):
                st.markdown(reponse.text)

    if user_picked == 'Image Captioning':
        model = gemini_vision()

        st.title("🖼️Image Captioning")

        image = st.file_uploader("Upload an image", type=[
                                 "jpg", "png", "jpeg"])

        user_prompt = st.text_input("Enter the prompt for image captioning:")

        if st.button("Generate Caption"):
            load_image = Image.open(image)

            colLeft, colRight = st.columns(2)

            with colLeft:
                st.image(load_image.resize((800, 500)))

            caption_response = gemini_visoin_response(
                model, user_prompt, load_image)

            with colRight:
                st.info(caption_response)

    if user_picked == 'Image editing':
        st.title("📸Photo Editing")
        image = st.file_uploader("Upload an image", type=[
                                 "jpg", "png", "jpeg"])

        if image:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)

            with col2:
                st.subheader("Edited Image")
                filter_type = st.selectbox('Choose an editing option:', [
                                           'Grayscale', 'Black and White', 'Pencil Sketch', 'Blur Effect'])

                if filter_type == 'Grayscale':
                    grayscale_image = convert_to_gray(Image.open(image))
                    if grayscale_image:
                        st.image(grayscale_image, use_column_width=True)
                    else:
                        st.error("Error converting image to grayscale.")

                elif filter_type == 'Black and White':
                    threshold = st.slider('Threshold:', 0, 255, 127)
                    bw_image = convert_to_black_and_white(
                        Image.open(image), threshold)
                    if bw_image:
                        st.image(bw_image, use_column_width=True)
                    else:
                        st.error("Error converting image to black and white.")

                elif filter_type == 'Pencil Sketch':
                    intensity = st.slider('Intensity:', 1, 255, 125)
                    sketch_image = convert_to_pencil_sketch(
                        Image.open(image), intensity)
                    if sketch_image:
                        st.image(sketch_image, use_column_width=True)
                    else:
                        st.error("Error converting image to pencil sketch.")

                elif filter_type == 'Blur Effect':
                    intensity = st.slider('Intensity:', 1, 100, 25)
                    blur_image = apply_blur_effect(
                        Image.open(image), intensity)
                    if blur_image:
                        st.image(blur_image, use_column_width=True)
                    else:
                        st.error("Error applying blur effect to the image.")
    

    if user_picked=="Generate Image":
        st.title("🪄Generated Image")
        with st.form(key="my_form"):
            query=st.text_area(
                label="Enter a prompt for image..",
                help="Enter a prompt for the image here..",
                key="query",
                max_chars=50,
            )
        
            submit_button = st.form_submit_button(label='Submit')

        if query and submit_button:
            with st.spinner(text="Generating image in progress..."):
                img_file=text2image(prompt=query)

            st.subheader("Your image")
            st.image(f"./{img_file}",caption=query)

    if user_picked == 'Health App':
        st.title("🏥Health App")
        
        # Input prompt for health analysis
        input_prompt = """
        You are an expert in nutritionist where you need to see the food items from the image
        and also suggest me that how much quantity should I eat and calculate the total calories, also provide the details of every food items with calories intake
        in the below format:

        1. Item 1 - no of calories
        2. Item 2 - no of calories
        ----
        ----

        Tell me some content that should I eat or not and what are the precautions should I take and give me suggestions for healthy food.Give me rating for the food out of 5.Bold the rating.
        
        """
        
        # Add your health app code here

        uploaded_file = st.file_uploader("Upload an image of food items...", type=["jpg", "jpeg", "png"])

        if st.button("Analyze Health"):
            if uploaded_file is not None:
                try:
                    image_data = input_image_setup(uploaded_file)
                    response = get_gemini_repsonse(input_prompt, image_data, input_prompt)
                    st.subheader("Health Analysis Result:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error analyzing health: {e}")
            else:
                st.warning("Please upload an image of food items.")
    
    if user_picked == 'Dehaze Effect':
        st.title("Image Dehazing App")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, -1)
            
            st.image(image, caption='Original Image', use_column_width=True)

            if st.button('Dehaze'):
                dehazed_image = dehaze_image(image)
                st.image(dehazed_image, caption='Dehazed Image', use_column_width=True)

if __name__ == "__main__":
    main()