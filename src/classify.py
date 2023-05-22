""" classify.py
Author: 
    Anton Drasbæk Schiønning (202008161), GitHub: @drasbaek

Desc:
    This script performs emotion classification on the titles of the fake news dataset using a HuggingFace model.
    By default, it uses the model 'j-hartmann/emotion-english-distilroberta-base', but it is possible to specify a different model using arguments.

Usage:
    $ python src/classify -m 'j-hartmann/emotion-english-distilroberta-base'
"""

# import packages
from transformers import pipeline
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

# suppress tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def arg_parse():
    '''
    Parse command line arguments.
    It is possible to specify the model to use for emotion classification.

    Returns:
    -   args (argparse.Namespace): Parsed arguments.

    '''
    
    # define parser
    parser = argparse.ArgumentParser(description="Classify emotions in titles.")

    # add argument
    parser.add_argument("--model", "-m", type=str, default="j-hartmann/emotion-english-distilroberta-base", help="HuggingFace Model to use for emotion classification.")

    # parse arguments
    args = parser.parse_args()

    return args


def define_paths():
    '''
    Define paths for input and output data.

    Returns:
    -   inpath (pathlib.PosixPath): Path to input data.
    -   outpath (pathlib.PosixPath): Path to where the classified data should be saved.
    '''

    # define paths
    path = Path(__file__)

    # define input dir
    inpath = path.parents[1] / "data" / "fake_or_real_news.csv"

    # define output dir
    outpath = path.parents[1] / "data"

    return inpath, outpath


def classify_emotion(data, classifier):
    '''
    Perform emotion classification on a dataframe of titles using a specified HuggingFace model for emotion classification.

    Args:
    -   data (pandas.DataFrame): Dataframe containing titles to classify (must have a column named "title").
    -   classifier (transformers.pipeline): HuggingFace pipeline for emotion classification.

    Returns:
    -   emotions (list): List of predicted emotions.
    -   scores (list): List of scores for predicted emotions.
    
    '''

    # create empty lists to store emotions and scores
    emotions = []
    scores = []

    # loop over all titles in data with a progress bar
    for title in tqdm(data["title"], desc="Classifying Headlines"):
    
        # predict primary emotion
        preds = classifier(title)

        # obtain emotion and its score
        emotion = preds[0][0]['label']
        score = preds[0][0]['score']

        # round score
        score = round(score, 2)

        # append both
        emotions.append(emotion)
        scores.append(score)
    
    return emotions, scores


def save_data(data, outpath, args):
    '''
    Saves classified data to a csv file, names based on the model used.

    Args:
    -   data (pandas.DataFrame): Dataframe containing classified data.
    -   outpath (pathlib.PosixPath): Path to where the classified data should be saved.
    -   args (argparse.Namespace): Parsed arguments.

    '''

    # abbreviate models to not include usernames
    if "/" in args.model:
        model_name = args.model.split("/")[1]
    else:
        model_name = args.model

    # define path
    path = outpath / 'classified_titles_{}.csv'.format(model_name)

    # write to csv
    data.to_csv(path, index=False)

    # print message
    print("Emotion Classification Complete. Results saved to {}".format(outpath / 'classified_titles_{}.csv'.format(model_name)))


def main():
    # define paths
    inpath, outpath = define_paths()

    # parse arguments
    args = arg_parse()

    # load data (only title and label columns)
    data = pd.read_csv(inpath, usecols=["title", "label"])

    # initialize classifier
    print("Initializing Classifier")
    classifier = pipeline("text-classification", 
                        model=args.model, 
                        return_all_scores=True,
                        top_k = 1)

    # run emotion classification
    emotions, scores = classify_emotion(data, classifier)

    # add to dataframe as columns
    data["predicted_emotion"] = emotions
    data["emotion_score"] = scores

    # save data
    save_data(data, outpath, args)


# run main
if __name__ == "__main__":
    main()