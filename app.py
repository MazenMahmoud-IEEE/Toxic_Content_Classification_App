# app.py
import streamlit as st
import pandas as pd
import os
import torch
from text_extractor import generate_caption
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from peft import PeftModel

# -------------------------
# Model setup
# -------------------------
BASE_MODEL = "distilroberta-base"
BEST_CHECKPOINT = "./checkpoint-66"

# Custom label mapping
id2label = {
    0: "Child Sexual Exploitation",
    1: "Elections",
    2: "Non-Violent Crimes",
    3: "Safe",
    4: "Sex-Related Crimes",
    5: "Suicide & Self-Harm",
    6: "Unknown S-Type",
    7: "Violent Crimes",
    8: "unsafe"
}
label2id = {v: k for k, v in id2label.items()}

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load base model architecture
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# Load LoRA adapter
lora_model = PeftModel.from_pretrained(
    base_model,
    BEST_CHECKPOINT,
    torch_dtype=torch.float32,
    device_map="cpu"  # force CPU
)

# Merge LoRA adapter into base model for inference
model = lora_model.merge_and_unload()

# Hugging Face pipeline for inference
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=-1  # CPU
)

# -------------------------
# Database setup
# -------------------------
DB_FILE = "database.csv"
if not os.path.exists(DB_FILE):
    df = pd.DataFrame(columns=["input_text_or_caption", "classification"])
    df.to_csv(DB_FILE, index=False)

# -------------------------
# Streamlit App
# -------------------------
st.title("🧠 Toxic Content Detection App")

option = st.radio("Choose input type:", ["Text", "Image"])

# Text classification
if option == "Text":
    user_text = st.text_area("Enter your text:")
    if st.button("Classify Text"):
        if user_text.strip() != "":
            result = classifier(user_text, top_k=1)[0]
            prediction = f"{result['label']} ({result['score']:.2f})"

            st.success(f"Classification: {prediction}")

            # Save to database
            df = pd.read_csv(DB_FILE)
            df = pd.concat([df, pd.DataFrame([[user_text, prediction]], columns=df.columns)],
                           ignore_index=True)
            df.to_csv(DB_FILE, index=False)
        else:
            st.warning("Please enter some text.")

# Image classification
elif option == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.button("Generate Caption & Classify"):
            # Generate caption
            caption = generate_caption(uploaded_image)
            st.info(f"Generated Caption: {caption}")

            # Run classification
            result = classifier(caption, top_k=1)[0]
            prediction = f"{result['label']} ({result['score']:.2f})"
            st.success(f"Classification: {prediction}")

            # Save to database
            df = pd.read_csv(DB_FILE)
            df = pd.concat([df, pd.DataFrame([[caption, prediction]], columns=df.columns)],
                           ignore_index=True)
            df.to_csv(DB_FILE, index=False)

# Database viewer
if st.checkbox("📂 View Database"):
    df = pd.read_csv(DB_FILE)
    st.dataframe(df)
