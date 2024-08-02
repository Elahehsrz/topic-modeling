from data import extract_all_text_as_list
from processing import preprocessing_texts
from tfidf import create_document_term_matrix
from lda import create_lsa_model, create_lda_model
from visualise import plot_word_cloud




filename = 'results.txt'
def main():
    num_topics = 5
    num_words = 20
    results = []
    print(type(results))
    pdf_path = "../Data/camus_la_chute.pdf"
    all_text_list = extract_all_text_as_list(pdf_path)
    text= all_text_list[0]
    procesed = preprocessing_texts(all_text_list)
    doc_term_matrix, dictionary = create_document_term_matrix(procesed)
    topics1 = create_lsa_model(doc_term_matrix, dictionary, num_topics, num_words)
    topics2 = create_lda_model(doc_term_matrix, dictionary, num_topics, num_words)
    plot_word_cloud(topics2, num_topics, num_words)


        

if __name__ == "__main__":
    main()

