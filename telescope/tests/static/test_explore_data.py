import pandas
from utils.explore_data import number_of_samples, number_of_samples_per_class
from utils.explore_data import frequency_of_classes_per_sample, median_of_words_per_sample


dataframe = pandas.DataFrame({'paragraph': ['Hello world!', 'Ola mundo!', 'Ciao mondo!'], 
            'category-a': [1, 1, 1],
            'category-b': [1, 1, 1],
            'category-c': [1, 1, 1]})

def test_number_of_samples():
    """Tests if the counting of samples if correct for a dummy dataframe
    """
    assert number_of_samples(dataframe) == 3

def test_number_of_samples_per_class():
    """Tests if the counting of samples per class is correct for a dummy dataframe
    """
    classes = ['category-a', 'category-b', 'category-c']
    samples_per_class = number_of_samples_per_class(dataframe, classes)

    for clss in classes:
        assert samples_per_class[clss] == 3

def test_frequency_of_classes_per_sample():
    """Tests if the generated frequency is correct for a dummy dataframe
    """
    classes = ['category-a', 'category-b', 'category-c']
    frequency = frequency_of_classes_per_sample(dataframe, classes)
    
    for number in frequency:
        assert frequency[number] == number

def test_median_of_words_per_sample():
    """Tests if the median of words is correct for a dummy dataframe
    """
    assert median_of_words_per_sample(dataframe) == 2.0
