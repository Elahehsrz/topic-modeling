import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.models.ldamodel import LdaModel



def parse_topic_string(topic_string):
    """Parse the topic string into a list of (word, weight) tuples."""
    return [(pair.split('*')[1].strip().strip('"'), float(pair.split('*')[0]))
            for pair in topic_string.split(' + ')]

def plot_word_cloud(topics, num_topics, num_words):
    """Generate word clouds for the given topics."""
    
    # Initialize the figure with a grid of subplots
    fig, axes = plt.subplots(1, num_topics, figsize=(20, 10), sharex=True, sharey=True)
    
    # Ensure the axes array is iterable even if num_topics is 1
    if num_topics == 1:
        axes = [axes]
    
    # Loop through each topic
    for i, (topic_num, topic_string) in enumerate(topics):
        # Parse the topic string to get word-weight pairs
        topic_words = parse_topic_string(topic_string)
        
        # Create a dictionary of the top words and their weights
        topic_dict = {word: weight for word, weight in topic_words[:num_words]}
        
        # Generate the word cloud
        wc = WordCloud(width=400, height=400, max_words=num_words, background_color='white')
        wc.generate_from_frequencies(topic_dict)
        
        # Plot the word cloud
        ax = axes[i]
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(f'Topic {topic_num}', fontsize=16)
        ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()