import streamlit as st
import pickle
import string
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

tfidf=pickle.load(open("vectorizer.pkl","rb"))
mnb=pickle.load(open("model.pkl","rb"))

st.title("SPAM or HAM !?")

input_sms = st.text_area("Enter your SMS...")

# preprocess(stemming,stop words etc.)
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]  # this is how you copy list
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))
    return " ".join(y)

if st.button("Predict", type="primary"):
    transformed_input_sms = transform_text(input_sms)

    # vectorize
    vectorised_input = tfidf.transform([transformed_input_sms])

    # predict
    prediction = mnb.predict(vectorised_input)[0]

    # display
    if prediction == 0:
        st.header("HAM")
    else:
        st.header("SPAM")