import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocessing_texts(documents):
    """
    Preprocess a list of documents by tokenizing, lowercasing, removing punctuation,
    removing stopwords, and lemmatizing.

    Parameters:
    documents (list of str): List of input text documents to preprocess.

    Returns:
    list of list of str: List of tokenized documents.
    """
    stop_words = set(stopwords.words('french'))
    lemmatizer = WordNetLemmatizer()

    preprocessed_documents = []
    for text in documents:
        # Tokenization
        tokens = word_tokenize(text)

        # Lowercasing
        tokens = [token.lower() for token in tokens]

        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]

        # Remove stopwords
        tokens = [token for token in tokens if token not in stop_words]

        # Lemmatization
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        preprocessed_documents.append(tokens)

    return preprocessed_documents









