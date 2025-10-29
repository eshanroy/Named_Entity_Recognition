from transformers import pipeline
import streamlit as st

# Load NER pipeline
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=False)

# Function to clean subword splits
def clean_entities(entities):
    cleaned = []
    current_word = ""
    current_label = ""
    current_score = 0.0
    for ent in entities:
        word = ent['word']
        label = ent['entity']
        score = ent['score']
        if word.startswith("##"):
            current_word += word[2:]
            current_score = max(current_score, score)
        else:
            if current_word:
                cleaned.append((current_word, current_label, round(current_score, 2)))
            current_word = word
            current_label = label
            current_score = score
    if current_word:
        cleaned.append((current_word, current_label, round(current_score, 2)))
    return cleaned

# Streamlit UI
st.set_page_config(page_title="NER Tagger", layout="centered")
st.title("🧠 Named Entity Recognition (NER) Tagger")
st.markdown("Enter a news headline or sentence to extract named entities like **Person**, **Location**, and **Organization**.")

# Input field
user_input = st.text_area("📝 Input Text", placeholder="e.g. Bihar Polls: Rahul Gandhi's ‘PM Modi Will Dance' Remark Fuels BJP Vs Congress")

# Run NER
if user_input:
    raw_entities = ner_pipeline(user_input)
    cleaned_entities = clean_entities(raw_entities)

    st.subheader("🔍 Extracted Entities (Cleaned)")
    for word, label, score in cleaned_entities:
        label_group = label.split("-")[-1]  # Convert B-PER to PER
        st.markdown(f"- **{word}** → `{label_group}` (confidence: {score})")