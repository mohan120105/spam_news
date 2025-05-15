import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit UI for user input
st.title("Fake News Detection")
st.write("""
    This app predicts whether a news article is **REAL** or **FAKE**.
    Simply paste the news text below and click "Predict".
""")

# User input text box
news_text = st.text_area("Enter news article text:")

# Predict when the button is pressed
if st.button('Predict'):
    if news_text.strip():
        # Transform input text to match the model's training format
        news_tfidf = vectorizer.transform([news_text])

        # Make prediction
        prediction = model.predict(news_tfidf)
        
        # Display result
        if prediction == 0:
            st.error("This is **FAKE** news!")
        else:
            st.success("This is **REAL** news!")
    else:
        st.warning("Please enter a news article text to predict.")
