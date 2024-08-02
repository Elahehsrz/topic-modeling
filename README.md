# Topic Modeling in "La Chute"

This repository contains a project for performing topic modeling on the text extracted from the PDF of Albert Camus' "La Chute". The project utilizes various natural language processing techniques to analyze the text and visualize the results through topic modeling.

## Project Overview

The project involves the following key steps:
1. **Data Extraction**: Extract text from each page of the PDF document.
2. **Text Processing**: Preprocess the text by tokenizing, lemmatizing, and removing stopwords.
3. **Feature Extraction**: Convert the cleaned text into a document-term matrix using TF-IDF.
4. **Topic Modeling**: Apply Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation (LDA) to identify topics.
5. **Visualization**: Generate word clouds for the identified topics to visualize the key terms.

## Directory Structure

- `data/`: Contains scripts for extracting text from PDF files.
- `processing/`: Contains scripts for text preprocessing.
- `tfidf/`: Contains scripts for creating the document-term matrix using TF-IDF.
- `lda/`: Contains scripts for generating LSA and LDA models.
- `visualise/`: Contains scripts for visualizing the topics using word clouds.
- `main.py`: The main script that ties all components together and runs the analysis.

## Installation

To run the project, ensure you have the following Python packages installed:

```bash
pip install PyPDF2 pymupdf spacy nltk gensim pyLDAvis matplotlib wordcloud
```

You will also need to download NLTK resources. You can do this by running the following commands in your Python environment:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage

1. **Extract Text**: Ensure you have the PDF file `camus_la_chute.pdf` in the `../Data/` directory.
2. **Run the Analysis**: Execute the `main.py` script to perform the analysis and generate visualizations.

```bash
python main.py
```

This script will extract text from the PDF, preprocess it, create a document-term matrix, perform topic modeling using LSA and LDA, and visualize the results with word clouds.

## Files

- **`data/extract_text_with_page_numbers`**: Extracts text from each page of the PDF and saves it in a JSON file.
- **`processing/preprocessing_texts`**: Preprocesses the extracted text by tokenizing, lemmatizing, and removing stopwords.
- **`tfidf/create_document_term_matrix`**: Creates a document-term matrix using TF-IDF.
- **`lda/create_lsa_model`, `lda/create_lda_model`**: Creates LSA and LDA models to identify topics.
- **`visualise/plot_word_cloud`**: Generates and plots word clouds for the identified topics.
- **`main.py`**: Main script to run the analysis.

## Results

The `results.txt` file will contain the output from the LSA and LDA models. Additionally, word clouds will be displayed for the topics identified by the LDA model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The text extraction methods use the PyMuPDF and PyPDF2 libraries.
- Natural language processing and topic modeling are performed using NLTK and Gensim.
- Visualization is done using Matplotlib and WordCloud.

