import pandas
import sklearn
import skmultilearn
from sklearn.ensemble import RandomForestClassifier
from classifier.prepare_data import shuffle_and_split, vectorize_paragraphs
from classifier.train_model import train

def test_classifier_training():
    """Tests if the training of classifiers is being performed correctly

    In our study, we use three different multiclass strategies to train the classifiers:
    Classifier Chain, One vs The Rest and Label Powerset. This method verifies if
    these strategies are being performed correctly, using as example the Random Forest
    classifier and a dummy dataframe.
    """
    classifier = RandomForestClassifier()
    dataframe = pandas.DataFrame({'paragraph': ['Hello world!', 'Ola mundo!', 'Ciao mondo!'], 
                'category-a': [1, 1, 1],
                'category-b': [1, 1, 1],
                'category-c': [1, 1, 1]})

    X_train, X_test, y_train, y_test = shuffle_and_split(
        dataframe, ['paragraph'], ['category-a', 'category-b', 'category-c'])
    X_train, X_test = vectorize_paragraphs(
        X_train['paragraph'].tolist(), X_test['paragraph'].tolist())

    training_args = {
        'classifier': classifier,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
    }

    strategy = 'classifier-chains'
    model, performance = train(strategy=strategy, **training_args)
    assert type(model) == skmultilearn.problem_transform.cc.ClassifierChain

    strategy = 'one-vs-the-rest'
    model, performance = train(strategy=strategy, **training_args)
    assert type(model) == sklearn.multiclass.OneVsRestClassifier

    strategy = 'label-powerset'
    model, performance = train(strategy=strategy, **training_args)
    assert type(model) == skmultilearn.problem_transform.lp.LabelPowerset