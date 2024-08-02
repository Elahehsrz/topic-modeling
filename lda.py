from gensim.models.ldamodel import LdaModel
from gensim import corpora
from gensim.models import LsiModel
from gensim.models import LdaModel

def create_lsa_model(doc_term_matrix, dictionary, num_topics, num_words):
    """
    Creates and prints an LSA model.

    Parameters:
    doc_term_matrix (list of lists): The document-term matrix.
    dictionary (gensim.corpora.Dictionary): The dictionary containing the id-to-word mapping.
    num_topics (int, optional): The number of topics to extract. Defaults to 3.
    num_words (int, optional): The number of words to print per topic. Defaults to 3.

    Returns:
    list: A list of topics with their corresponding weights and words.
    """
    # Create the LSA model
    lsa = LsiModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary)
    
    # Print the topics
    topics = lsa.print_topics(num_topics=num_topics, num_words=num_words)
    for topic in topics:
        print(topic)
    
    return topics



def create_lda_model(doc_term_matrix, dictionary, num_topics, num_words):
    """
    Creates and prints an LDA model.

    Parameters:
    doc_term_matrix (list of lists): The document-term matrix.
    dictionary (gensim.corpora.Dictionary): The dictionary containing the id-to-word mapping.
    num_topics (int, optional): The number of topics to extract. Defaults to 3.
    num_words (int, optional): The number of words to print per topic. Defaults to 3.

    Returns:
    list: A list of topics with their corresponding weights and words.
    """
    # Create the LDA model
    lda = LdaModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary)
    
    # Print the topics
    topics = lda.print_topics(num_topics=num_topics, num_words=num_words)
    for topic in topics:
        print(topic)
    
    return topics









