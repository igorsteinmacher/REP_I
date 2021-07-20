import os
import pandas
# Heuristic
from spacy.lang.en import English
from sklearn.feature_selection import VarianceThreshold
# Statistic
from sklearn.feature_extraction.text import TfidfVectorizer

def create_statistic_features(X_train, X_test):
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
    train_statistic_features = vectorizer.fit_transform(X_train)
    test_statistic_features = vectorizer.transform(X_test)

    return train_statistic_features, test_statistic_features

def create_heuristic_features(X_train, X_test):
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
    jsonl_filepath = os.path.join(os.getcwd(), 'data_preparation', 'patterns.jsonl')
    ruler = nlp.add_pipe("entity_ruler").from_disk(jsonl_filepath)

    train_heuristic_features = pandas.DataFrame()
    train_heuristic_features['Paragraph'] = X_train

    test_heuristic_features = pandas.DataFrame()
    test_heuristic_features['Paragraph'] = X_test

    for heuristic in ruler.patterns:
        train_heuristic_features[heuristic['id']] = 0
        test_heuristic_features[heuristic['id']] = 0

    for index, row in train_heuristic_features.iterrows():
        doc = nlp(row['Paragraph'])

        for heuristic in doc.ents:
            train_heuristic_features.at[index, heuristic.ent_id_] = 1

    for index, row in test_heuristic_features.iterrows():
        doc = nlp(row['Paragraph'])

        for heuristic in doc.ents:
            test_heuristic_features.at[index, heuristic.ent_id_] = 1

    train_heuristic_features.drop('Paragraph', axis=1, inplace=True)
    test_heuristic_features.drop('Paragraph', axis=1, inplace=True)

    # The VarianceThreshold is a feature selection algorithm that
    # removes all low-variance features. Features with variance
    # lower than 0.95 are removed in this study. This threshold 
    # was manually selected.
    # Learn more at: scikit-learn.org/stable/modules/feature_selection.html
    selector = VarianceThreshold(threshold=(.95 * (1 - .95)))
    train_heuristic_features = selector.fit_transform(train_heuristic_features)
    test_heuristic_features = selector.transform(test_heuristic_features)


    return train_heuristic_features, test_heuristic_features