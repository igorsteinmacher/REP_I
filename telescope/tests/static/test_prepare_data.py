import scipy
import pandas
from classifier.prepare_data import shuffle_and_split, vectorize_paragraphs

def test_shuffle_and_split():
    """Tests if `shuffle_and_split` is correctly preparing the training and test sets 

    After reading the dataframe of paragraphs, the `shuffle_and_split` method
    must shuffle and divide the dataframe in training in test sets. 80 per cent
    of the dataframe must be dedicated for the training set, and 20 per cent for
    the test set.
    """
    paragraphs = ['Hello world!', 'Ola mundo!', 'Ciao mondo!', 'Здраво светот!', 'Hallo Welt!']

    dataframe = pandas.DataFrame({'paragraph': paragraphs, 
            'category-a': [1, 1, 1, 1, 1],
            'category-b': [1, 1, 1, 1, 1],
            'category-c': [1, 1, 1, 1, 1]})
    
    features = ['paragraph']
    classes = ['category-a', 'category-b', 'category-c']
    X_train, X_test, y_train, y_test = shuffle_and_split(dataframe, features, classes)

    assert len(X_train) == 4
    assert len(y_train) == 4
    assert len(X_test) == 1
    assert len(y_test) == 1

def test_paragraphs_vectorization():
    """Tests if the vectorization of paragraphs is being made by `vectorize_paragraphs`

    Besides dividing the dataframe in training and test sets, the classifier package
    must also transform the paragraphs into acceptable values for classification.
    For this reason, the `vectorize_paragraphs` exists, it transforms the paragraphs
    strings into a matrix of TF-IDF features. This testing method checks whether
    the test and training matrix is ​​being performed.
    """
    paragraphs = ['Hello world!', 'Ola mundo!', 'Ciao mondo!', 'Здраво светот!', 'Hallo Welt!']

    dataframe = pandas.DataFrame({'paragraph': paragraphs, 
            'category-a': [1, 1, 1, 1, 1],
            'category-b': [1, 1, 1, 1, 1],
            'category-c': [1, 1, 1, 1, 1]})
    
    features = ['paragraph']
    classes = ['category-a', 'category-b', 'category-c']
    X_train, X_test, y_train, y_test = shuffle_and_split(dataframe, features, classes)
    X_train, X_test = vectorize_paragraphs(X_train, X_test)

    assert type(X_train) == scipy.sparse.csr.csr_matrix
    assert type(X_test) == scipy.sparse.csr.csr_matrix
