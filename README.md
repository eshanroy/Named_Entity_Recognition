# Named_Entity_Recognition
A beginner-friendly NLP project that extracts named entities (Person, Location, Organization) from news headlines or sentences using Hugging Face Transformers. Built with Streamlit for interactive input and cleaned output using subword handling.

# Features
- Uses Hugging Face’s dslim/bert-base-NER model
- Cleans subword tokens like Ra, ##hul into readable names
- Displays entity type and confidence score
- Streamlit UI for easy interaction
- Ideal for NLP beginners, exam demos, and AI portfolios

# Tech Stack
- Python
- Hugging Face Transformers
- Streamlit

# Installation
pip install transformers torch streamlit

# How to Run
streamlit run ner_clean_tagger.py

# Sample Input 
Virat Kohli scores century in Bengaluru

# Sample Output
Extracted Entities
Virat → PER (confidence: 1.0)
Kohli → PER (confidence: 1.0)
Bengaluru → LOC (confidence: 1.0)
