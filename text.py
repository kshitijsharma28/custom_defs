import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords


def text_cleanup(text_col):
    # Define variables for replacing to get cleaned text
    sr = stopwords.words('english')
    short_forms = ["i'm", "you're"]
    punctuations = ["'", "!", "#", "."]

    # Convert to lower characters
    output_col = text_col.str.lower()
    
    # Removing couple of short forms
    output_col = output_col.apply(lambda x: \
      ' '.join([y if y not in short_forms else '' for y in x.split(' ')]))
        
    # Removing stopwords
    output_col = output_col.apply(lambda x: \
      ' '.join([y if y not in sr else '' for y in x.split(' ')]))
    
    # Removing punctuations
    for symbol in punctuations:
        output_col = output_col.str.replace(symbol, '')
            
    # Removing leading and trailing spaces
    output_col = output_col.str.strip()
    
    return output_col
