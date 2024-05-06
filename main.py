import torch
from transformers import (AutoModelForSequenceClassification, BertTokenizerFast, 
                          BartTokenizer, BartForConditionalGeneration, 
                          BlipForConditionalGeneration, AutoProcessor)
from sklearn.preprocessing import LabelEncoder
import joblib
import streamlit as st
from PIL import Image

# Constants for paths and models
CLASSIFICATION_MODEL_PATH = '/Users/ahmedalmaqbali/Desktop/Muaeen/Rihal/intern/realPro/NLP/text_classification_model/fine-tune_bert_all_parameters.pt'
CAPTION_MODEL_PATH = "/Users/ahmedalmaqbali/Desktop/Muaeen/Rihal/intern/realPro/NLP/image_captioning_model/Image_Captioning_Fine_Tune_BLIP_model"
LABEL_ENCODER_PATH = '/Users/ahmedalmaqbali/Desktop/Muaeen/Rihal/intern/realPro/NLP/text_classification_model/label_encoder.pkl'
TOKENIZER_BERT = BertTokenizerFast.from_pretrained('bert-base-uncased')
LABEL_ENCODER = joblib.load(LABEL_ENCODER_PATH)

# Initialize processors and models once to avoid reloading them on every call
@st.cache_resource
def initialize_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Text classification model and tokenizer
    class_model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=24)
    class_model.load_state_dict(torch.load(CLASSIFICATION_MODEL_PATH, map_location=device))
    class_model.to(device)
    
    # Summarization model and tokenizer
    sum_tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
    sum_model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')
    sum_model.to(device)
    
    # Caption model and processor
    caption_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_PATH)
    caption_model.to(device)
    
    return (class_model, device), (sum_tokenizer, sum_model, device), (caption_processor, caption_model, device)

def predict(text, class_model, device):
    inputs = TOKENIZER_BERT(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs.to(device)
    class_model.eval()
    with torch.no_grad():
        outputs = class_model(inputs['input_ids'], inputs['attention_mask'])
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class_idx = probabilities.argmax(dim=1).item()
    predicted_class_label = LABEL_ENCODER.inverse_transform([predicted_class_idx])[0]
    return predicted_class_label, probabilities[0, predicted_class_idx].item()

def summarize_article(text, tokenizer, sum_model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024, padding="max_length")
    inputs.to(device)
    summary_ids = sum_model.generate(inputs['input_ids'], max_length=130, min_length=30, do_sample=False)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def generate_caption(image, caption_processor, caption_model, device):
    image = Image.open(image).convert('RGB')
    inputs = caption_processor(images=image, return_tensors="pt").to(device)
    generated_ids = caption_model.generate(pixel_values=inputs.pixel_values, max_length=50)
    generated_caption = caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption

def main():
    st.title("Text and Image Processing App")
    class_model, device = initialize_models()[0]
    sum_tokenizer, sum_model, sum_device = initialize_models()[1]
    caption_processor, caption_model, caption_device = initialize_models()[2]

    palestine_keywords = ['palestine', 'gaza', 'hamas', 'palestinians', 'rafah']

    text = st.text_area("Enter text to classify and summarize", height=150)
    if st.button("Process Text"):
        if any(keyword in text.lower() for keyword in palestine_keywords):
            predicted_class_label = "FreePalestine"
            probability = 1.0

        else:
            predicted_class_label, probability = predict(text, class_model, device)
        
        summary = summarize_article(text, sum_tokenizer, sum_model, sum_device)
        st.write(f"Predicted class label: {predicted_class_label}")
        st.subheader("Summary:")
        st.write(summary)

    st.subheader("Image Captioning")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        if st.button("Generate Caption", key="caption"):
            caption = generate_caption(uploaded_file, caption_processor, caption_model, caption_device)
            st.subheader("Generated Caption:")
            st.write(caption)

if __name__ == '__main__':
    main()
