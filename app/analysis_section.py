#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import count
import pandas
import collections
import plotly.express as px
from annotated_text import annotated_text
from classifier.classify_content import get_contributing_predictions

classes = {'No categories identified.': "#577590",
    'CF – Contribution flow': "#f94144",
    'CT – Choose a task': "#f3722c",
    'TC – Talk to the community': "#f8961e",
    'BW – Build local workspace': "#f9c74f",
    'DC – Deal with the code': "#90be6d",
    'SC – Submit the changes': "#43aa8b"}

def write_overview_reasoning(page, predictions_per_class):
    # Ignore the class "No categories identified"
    predictions_per_class = predictions_per_class[predictions_per_class['Category'] != 'No categories identified.']
    percentage_existent_categories = (predictions_per_class['Number of paragraphs'] > 0).sum() / predictions_per_class['Number of paragraphs'].count() * 100

    if percentage_existent_categories == 100:
        contributing_quality = 'Excellent'
    elif percentage_existent_categories > 83:
        contributing_quality = 'Very Good'
    elif percentage_existent_categories > 66:
        contributing_quality = 'Good'
    elif percentage_existent_categories > 50:
        contributing_quality = 'Regular'
    elif percentage_existent_categories > 33:
        contributing_quality = 'Poor'
    else:
        contributing_quality = 'Very Poor'

    quality_reasonings = {
        'Excellent': 'It means that your documentation file covered all the categories of information a newcomer usually needs to known before attempting to join in an open source project. \
            Please read our analysis below to make it even better.',
        'Very Good': 'It means we identified five out of six categories of information known to be relevant for newcomers in your documentation. To make your project more receptive for new\
            contributors, make sure that your document covers all the six categories of information. We provide a more detailed analysis about your documentation file below.',
        'Good': 'It means that four out of six categories of information known to be relevant for newcomers were identified in your documentation file. We believe that we can help you make it even better.\
            Please, take a look at our detailed analysis below and improve your file when you judge necessary.',
        'Regular': 'Only half of the categories of information known to be relevant for newcomers were identified in your documentation file. To guarantee a better retention of new contributors,\
                make sure your documentation file is covering the six categories of information we describe below. We known you make it better!',
        'Poor': 'Unfortunately, only two categories of information known to be relevant for newcomers in open-source projects was identified. To make sure this is not a classification misunderstanding, please\
            review the evaluation we describe below. Let\'s make your project more receptive!',
        'Very Poor': "Less than two categories of information known to be relevant for newcomers were identified in your documentation file. To make sure that this is not a problem with our prediction, please\
            read the review below and make changes to your file that you judge as necessary. We hope you can make your project even better!"    
    }

    page.markdown("According to our classification model, the quality of your CONTRIBUTING file is **{}**. {}".format(contributing_quality, quality_reasonings[contributing_quality]))

def write_overview_barplot(page, predictions_per_class):
    fig = px.bar(predictions_per_class, x = "Number of paragraphs", y = "Repository", color = "Category", color_discrete_map=classes, orientation = "h", height=170, template = 'seaborn')
    fig.update_layout(yaxis = {'visible': False, 'showticklabels': False}, legend={'title_text': None}, paper_bgcolor='rgb(245, 245, 245)')
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1))

    # Hide legend for categories where number of paragraphs is zero
    for trace in fig['data']:
        if predictions_per_class.loc[predictions_per_class['Category'] == trace['name'], 'Number of paragraphs'].item() == 0:
            trace['showlegend'] = False

    page.plotly_chart(fig, use_container_width = True)

def write_annotated_paragraphs(page, paragraphs, predicted_categories):
    with page.expander("Open document with predictions"):
        for paragraph, predicted_category in zip(paragraphs, predicted_categories):
            if predicted_category == 'No categories identified.':
                page.write(paragraph)
            if predicted_category.startswith('CT'):
                annotated_text((paragraph, predicted_category, classes[predicted_category]))
            if predicted_category.startswith('CF'):
                annotated_text((paragraph, predicted_category, classes[predicted_category]))
            if predicted_category.startswith('TC'):
                annotated_text((paragraph, predicted_category, classes[predicted_category]))
            if predicted_category.startswith('BW'):
                annotated_text((paragraph, predicted_category, classes[predicted_category]))
            if predicted_category.startswith('DC'):
                annotated_text((paragraph, predicted_category, classes[predicted_category]))
            if predicted_category.startswith('SC'):
                annotated_text((paragraph, predicted_category, classes[predicted_category]))

def count_predictions_per_class(predictions, repository_url):

    predictions_per_class = pandas.DataFrame.from_dict(collections.Counter(predictions), orient='index', columns=['Number of paragraphs']).reset_index()
    predictions_per_class = predictions_per_class.rename(columns = {"index": "Category"})
    predictions_per_class['Repository'] = repository_url

    for category in classes.keys():
        if category not in predictions_per_class['Category'].values:
            predictions_per_class.loc[len(predictions_per_class.index), ['Category', 'Number of paragraphs', 'Repository', 'Color']] = [category, 0, repository_url, classes[category]]
        else:
            predictions_per_class.loc[predictions_per_class['Category'] == category, ['Color']] = classes[category]

    print(predictions_per_class)
    print(classes)
    return predictions_per_class

def write_contributing_analysis(page, repository_url):
    paragraphs, predictions = get_contributing_predictions(repository_url)

    if len(paragraphs) > 0 and len(predictions) > 0:
        predictions_per_class = count_predictions_per_class(predictions, repository_url)
        page.write("#### How good is my documentation file?")
        write_overview_barplot(page, predictions_per_class)
        write_overview_reasoning(page, predictions_per_class)
        write_annotated_paragraphs(page, paragraphs, predictions)
        # Strong points
        # Weak points
