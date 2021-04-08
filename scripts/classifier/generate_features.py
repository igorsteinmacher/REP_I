import pandas
# Heuristic
from spacy.lang.en import English
from sklearn.feature_selection import VarianceThreshold
# Statistic
from sklearn.feature_extraction.text import TfidfVectorizer

def create_statistic_features(X):
    """Converts raw training and test paragraphs to TF-IDF features.

    Args:
        X: A sequence of paragraphs for training.

    Returns:
        A matrix of TF-IDF features for training and a matrix of TF-IDF features
        for test.
    """

    vect_args = {
        'ngram_range': (1, 2),  # Google recomends: 1-gram + 2-grams
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'stop_words': 'english',
        'analyzer': 'word',
    }
    
    vectorizer = TfidfVectorizer(**vect_args)
    statistic_features = vectorizer.fit_transform(X)

    return statistic_features

def create_heuristic_features(X):
    nlp = English()
    ruler = nlp.add_pipe("entity_ruler").from_disk("./patterns.jsonl")

    heuristic_features = pandas.DataFrame()
    heuristic_features['Paragraph'] = X

    for heuristic in ruler.patterns:
        heuristic_features[heuristic['id']] = 0

    for index, row in heuristic_features.iterrows():
        doc = nlp(row['Paragraph'])

        for heuristic in doc.ents:
            heuristic_features.at[index, heuristic.ent_id_] = 1

    heuristic_features.drop('Paragraph', axis=1, inplace=True)
    selector = VarianceThreshold(threshold=(.95 * (1 - .95)))
    heuristic_features = selector.fit_transform(heuristic_features)

    return heuristic_features