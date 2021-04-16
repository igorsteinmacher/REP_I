import pandas
# Heuristic
from spacy.lang.en import English
from sklearn.feature_selection import VarianceThreshold
# Statistic
from sklearn.feature_extraction.text import TfidfVectorizer

def create_statistic_features(X):
    """Converts paragraphs into TF-IDF features.

    Note that in this study, the TF-IDF features are mentioned
    as statistic features, while rule-based features are grouped
    as heuristic features.

    Args:
        X: A string column containing paragraphs.

    Returns:
        A sparse matrix of TF-IDF features.
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
    """Creates a set of features using a rule-based matching approach over paragraphs.

    To improve the performance of the classification models, a set of rule-based features were
    created in conjunction with the TF-IDF features.

    Each rule defines a new feature in X, and it is represented as a column in the dataframe. 
    Each rule was manually defined by the researchers based on what they have learned during the qualitative analysis.
    The row values of each rule (column) are defined based on the expression given by the respective rule. 
    There is, for example, a rule in this study that verifies if the word "GitHub" appears in each paragraph.
    If the word appears in a paragraph, it defines the row value of the rule's column as 1 or 0 otherwise.

    All rules of this study are presented in the `patterns.jsonl` file inside the `classifier` folder.
    Learn more about rule-based matching at: spacy.io/usage/rule-based-matching

    Args:
        X: A string column containing paragraphs.

    Returns:
        A sparse matrix of heuristic features.
    """

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

    # The VarianceThreshold is a feature selection algorithm that
    # removes all low-variance features. Features with variance
    # lower than 0.95 are removed in this study. This threshold 
    # was manually selected.
    # Learn more at: scikit-learn.org/stable/modules/feature_selection.html
    selector = VarianceThreshold(threshold=(.95 * (1 - .95)))
    heuristic_features = selector.fit_transform(heuristic_features)

    return heuristic_features