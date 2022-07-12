import pickle
from classifier.get_contributing import get_contributing_file
from classifier.get_features import convert_paragraphs_into_features

def get_contributing_predictions(repository_url):
    paragraphs = get_contributing_file(repository_url)

    if paragraphs:
        # Loads the classification model.
        model = pickle.load(open('classifier/classification_model.sav', 'rb'))

        # Using the estimator, predicts the classes for the paragraphs in the file
        predictions = model.predict(convert_paragraphs_into_features(paragraphs))

        return paragraphs, predictions

    return [], []
