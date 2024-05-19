import streamlit as st
from transformers import pipeline
from PIL import Image
import google.generativeai as genai

# Gemini AI API-Key
api_key = "AIzaSyCXZz46qd6y5wkSqPSKhGqlXvQVIni_GqY"


# Initialize the image classification pipeline
pipe = pipeline("image-classification", model="jazzmacedo/fruits-and-vegetables-detector-36")

# Streamlit app
st.title("Fruits and Vegetables Detector")
st.write("Upload an image of a fruit or vegetable to get a prediction.")

with st.form("my_form"):
# File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the image file
        image = Image.open(uploaded_file)

        # Display the image
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        
        # Make prediction
    clicked = st.form_submit_button("Predict")
    if clicked:
        with st.spinner("Classifying..."):
            prediction = pipe(image)
        # Display the prediction
        st.write("Prediction:")
        st.success(f"Predicted {prediction[0]['label']} with {100*prediction[0]['score']:.2f}% score")  
    
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-pro")
        prompt = f"Provide detailed information about the fruit or vegetable '{prediction[0]['label']}'. Include its physical characteristics, nutritional benefits, common culinary uses, growing conditions, and any interesting historical or cultural facts."                
        # Adding a spinner to make sense while generating details
        with st.spinner(f"Generating information about {prediction[0]['label']}..."):
            response = model.generate_content(prompt)
            if response:
                            
                            # Writing response to the UI
                st.write(response.text)
            else:
                st.error("Failed to generate information. Please try again later.")