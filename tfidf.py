from gensim import corpora
from gensim.models import TfidfModel


def create_document_term_matrix(clean_corpus):
    """
    Create a document-term matrix from a cleaned corpus.

    Parameters:
    clean_corpus (list of list of str): List of tokenized documents.

    Returns:
    doc_term_matrix (list of list of tuple): Document-term matrix where each element is a (word_id, word_frequency) tuple.
    dictionary (gensim.corpora.Dictionary): Dictionary mapping words to their unique ids.
    """
    # Create dictionary
    dictionary = corpora.Dictionary(clean_corpus)
    
    # Create document-term matrix
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_corpus]
    
    return doc_term_matrix, dictionary



