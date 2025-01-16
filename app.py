import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

# Preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # Apply stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("Email Spam Classifier")
input_email = st.text_area("Enter the message")

if st.button('Predict'):
    if input_email.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # Preprocess input
        transformed_email = transform_text(input_email)
        # Vectorize input
        vector_input = tfidf.transform([transformed_email])
        # Predict
        result = model.predict(vector_input)[0]
        # Display result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")

# to run app write in terminal
# streamlit run app.py
