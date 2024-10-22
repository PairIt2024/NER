import re
import spacy

nlp = spacy.load('en_core_web_sm')

def preprocess_and_tokenize(text):
    #Lowercase and no special characters
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    #Tokenize the text
    doc = nlp(text)
    tokens = [token.text for token in doc]
    #Return tokens
    return tokens


