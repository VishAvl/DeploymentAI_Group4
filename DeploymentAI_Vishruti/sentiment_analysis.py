import streamlit as st
import pickle

# Load the saved vectorizer and model
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

with open('naive_model.pkl', 'rb') as file:
    sentiment_model = pickle.load(file)

# Streamlit app
st.title('Sentiment Analysis App')

# Input text from the user
user_input = st.text_area('Enter a sentence to analyze sentiment')

# Predict sentiment
if st.button('Analyze Sentiment'):
    try:
        # Transform the input using the loaded vectorizer
        user_input_vectorized = vectorizer.transform([user_input])

        # Predict sentiment
        prediction = sentiment_model.predict(user_input_vectorized)
        
        # Map prediction to sentiment label
        sentiment_label = 'Positive' if prediction[0] == 1 else 'Negative' if prediction[0] == 0 else 'Neutral'
        
        # Display the sentiment
        st.write(f'The predicted sentiment is: {sentiment_label}')
    except Exception as e:
        st.error(f"An error occurred: {e}")
