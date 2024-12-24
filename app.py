import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
import docx
import os

# Ensure necessary resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean the text
def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

# Function to read text from a .docx file
def read_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)

# Function to generate summary using the summarization model
def generate_summary(text, tokenizer, model, max_length=1500, min_length=100, max_input_length=512):  # Adjusted lengths
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=1.5,  # Adjusted to favor longer summaries
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function to answer a question based on the context text
def answer_question(context, question, tokenizer, model):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        answer_start = outputs.start_logits.argmax()
        answer_end = outputs.end_logits.argmax()
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start: answer_end + 1]))
    return answer

# Load summarization model and tokenizer (cached)
@st.cache_resource
def load_summarization_model():
    model_name = "google/pegasus-xsum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Load question-answering model and tokenizer (cached)
@st.cache_resource
def load_qa_model():
    qa_model_name = "deepset/roberta-base-squad2"
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    return qa_tokenizer, qa_model

# Streamlit app layout
st.sidebar.title("Select Functionality")
option = st.sidebar.selectbox("Choose an option:", ["Summarize Document", "Answer Questions", "Find Similar Cases"])

if option == "Summarize Document":
    st.title("Document Summarization")
    uploaded_file = st.file_uploader("Upload a Word document (.docx) for summarization", type="docx")
    if uploaded_file:
        text = read_text_from_docx(uploaded_file)
        st.subheader("Original Document Text:")
        st.text(text)
        
        # Load models and generate summary
        tokenizer, model = load_summarization_model()
        st.write("Model loaded successfully.")
        
        # Generate summary with progress updates
        st.write("Generating summary...")
        summary = generate_summary(text, tokenizer, model)
        
        st.subheader("Generated Summary:")
        st.write(summary)

elif option == "Answer Questions":
    st.title("Question Answering")
    uploaded_file = st.file_uploader("Upload a Word document (.docx) as context", type="docx")
    question_text = st.text_input("Enter your question:")
    if uploaded_file and question_text:
        context_text = read_text_from_docx(uploaded_file)
        st.subheader("Context:")
        st.write(context_text)
        
        # Load QA model and generate answer
        qa_tokenizer, qa_model = load_qa_model()
        st.write("QA model loaded successfully.")
        
        answer = answer_question(context_text, question_text, qa_tokenizer, qa_model)
        
        st.subheader("Answer:")
        st.write(answer)

elif option == "Find Similar Cases":
    st.title("Find Similar Cases")
    
    # Automatically load the caseLaw.csv file from the local directory
    case_law_path = os.path.join(os.getcwd(), 'caselaws.csv')
    
    # Check if the CSV exists in the current directory
    if os.path.exists(case_law_path):
        df = pd.read_csv(case_law_path)
        st.write("Case law database loaded successfully.")
        
        # Check if 'Summary' column exists
        if 'Summary' not in df.columns:
            st.error("The CSV file must contain a 'Summary' column.")
        else:
            df['Cleaned_Summary'] = df['Summary'].apply(clean_text)
        
            # File uploader for the query document (Word doc)
            query_file = st.file_uploader("Upload a Word document (.docx) with the summary for similarity comparison", type="docx")
            
            if query_file:
                # Read and clean the uploaded query document
                query_text = read_text_from_docx(query_file)
                cleaned_query = clean_text(query_text)
        
                # Combine all cleaned summaries with the query for TF-IDF calculation
                summaries = df['Cleaned_Summary'].tolist() + [cleaned_query]
        
                # Create a TF-IDF vectorizer
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(summaries)
        
                # Extract the TF-IDF matrix for the query
                query_vector = tfidf_matrix[-1]  # The last vector corresponds to the query
        
                # Calculate cosine similarity between the query and each case summary
                similarity_scores = cosine_similarity(query_vector, tfidf_matrix[:-1]).flatten()
        
                # Add the similarity scores to the DataFrame
                df['Similarity_Score'] = similarity_scores
        
                # Check if there are any similar cases with a non-zero score
                if df['Similarity_Score'].max() > 0:
                    # Retrieve the top 5 most similar cases based on highest similarity scores
                    top_5_similar_cases = df.nlargest(5, 'Similarity_Score')
        
                    st.subheader("Top 5 Most Similar Cases to Uploaded Summary:")
                    for index, row in top_5_similar_cases.iterrows():
                        st.write(f"\n**Case {index + 1}:**")
                        st.write(f"**Summary:** {row['Summary']}")
                        st.write(f"**Similarity Score:** {row['Similarity_Score']}")
                else:
                    st.warning("No similar cases found with a significant similarity score.")
    else:
        st.error("caseLaw.csv file not found in the current directory.")
