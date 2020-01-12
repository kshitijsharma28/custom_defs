import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
import nltk

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


def text_cleanup(text_col):
    # Define variables for replacing to get cleaned text
    sr = stopwords.words('english')
    
    short_forms = {"i'm"       : 'i am',
                   "don't"     : 'do not',
                   "magic's"   : 'magic is',
                   "she's"     : 'she is',
                   "should've" : 'should have',
                   "where's"   : 'where is',
                   "would've"  : 'would have',
                   "you're"    : 'you are'}
    
    punctuations = ["'", "!", "#", ".", "?", '/', '\\', '@', '_', '-', '(', ')', '%20']
                    
    ignore_patterns = ['&lt;', '&gt;', '&amp;']

    # Convert to lower characters
    output_col = text_col.str.lower()
    
    # Replacing links
    output_col = output_col.replace(r'https?://[\d\w\./]+', ' link ', regex=True)
    
    # Replacing @ tags
    output_col = output_col.replace('(@[\w_\d]+)', ' taggedword ', regex=True)
    
    # Replacing # tags
    output_col = output_col.replace('(#[\w_\d]+)', ' hashtag ', regex=True)
    
    # Replacing short forms
    output_col = output_col.apply(lambda x: \
      ' '.join([y if y not in short_forms else '' for y in x.split(' ')]))
                    
    #Removing specific words
    for pattern in ignore_patterns:
        output_col = output_col.replace(pattern, ' ', regex=False)
        
    # Removing stopwords
    output_col = output_col.apply(lambda x: \
      ' '.join([y if y not in sr else '' for y in x.split(' ')]))
    
    # Removing punctuations
    for symbol in punctuations:
        output_col = output_col.str.replace(symbol, ' ')

    # Lemmatisation
    def lemmatize_text(text):
        return ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])

    output_col = output_col.apply(lemmatize_text)
        
    # Replacing multiple & leading / trailing spaces
    output_col = output_col.replace(r'\s{2,}', ' ', regex=True).str.strip()
            
    return output_col
