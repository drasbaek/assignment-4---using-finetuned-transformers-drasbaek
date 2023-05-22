""" visualize.py
Author: 
    Anton Drasbæk Schiønning (202008161), GitHub: @drasbaek

Desc:
    This script creates visualizations of the classification results, produced in classify.py and found in the data folder.
    Three types of outputs are created:
    -   classification_overview.csv: A summary of the classification results.
    -   emotion_distribution.png: A barplot of the distribution of emotions in the dataset.
    -   emotions_by_label.png: Pie charts of the distribution of emotions in the dataset, split by label.

Usage:
    $ python src/visualize -m 'j-hartmann/emotion-english-distilroberta-base'
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os.path
import requests
from pathlib import Path
import seaborn as sns
import argparse

def arg_parse():
    '''
    Parse command line arguments.
    It is possible to specify the model folder to base visualizations on.
    Returns:
    -   args (argparse.Namespace): Parsed arguments.

    '''

    # create parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--model", "-m", type=str, default="j-hartmann/emotion-english-distilroberta-base", help="Name of the classifier to use.")

    # parse arguments
    args = parser.parse_args()

    return args


def define_paths(args):
    '''
    Define paths for input and output data.

    Args:
    -   args (argparse.Namespace): Parsed arguments.

    Returns:
    -   inpath (pathlib.PosixPath): Path to input data.
    -   outpath (pathlib.PosixPath): Path to output data.
    '''

    # define paths
    path = Path(__file__)

    # abbreviate models to not include usernames
    if "/" in args.model:
        model_name = args.model.split("/")[1]
    else:
        model_name = args.model

    # define input dir
    inpath = path.parents[1] / "data" / "classified_titles_{}.csv".format(model_name)

    # define output dir
    outpath = path.parents[1] / "out" / "results_{}".format(model_name)

    # create output dir if not exists
    Path(outpath).mkdir(parents=True, exist_ok=True)

    return inpath, outpath


def get_clf_summary(data, outpath):
    """
    Pivots data to create a summary of the classification results.

    Args:
    -   data (pandas.DataFrame): Dataframe with the classification results (output from classify.py)
    -   outpath (pathlib.PosixPath): Path to output data.

    Returns:
    -   pivot_data (pandas.DataFrame): Dataframe with the summary of the classification results.
    """

    # pivot table to summarize counts by emotion and label
    pivot_data = data.pivot_table(index='predicted_emotion', columns='label', aggfunc='size', fill_value=0)

    # Reset index and rename columns
    pivot_data = pivot_data.reset_index().rename(columns={'predicted_emotion': 'predicted_emotion', 'FAKE': 'fake_only', 'REAL': 'real_only'})

    # add a column for total headlines
    pivot_data["all_headlines"] = pivot_data["fake_only"] + pivot_data["real_only"]

    # reorder columns
    pivot_data = pivot_data[['predicted_emotion', 'all_headlines', 'real_only', 'fake_only']]

    # write to csv
    pivot_data.to_csv(outpath / 'classification_overview.csv', index=False)

    return pivot_data


def add_rose_pine_styles(overwrite: bool=False):
    '''
    Adds the rose-pine styles to the matplotlib style library. 
    Code taken directly from https://levelup.gitconnected.com/ros%C3%A9-pine-elegant-matplotlib-theme-for-crisp-plots-df40c6bf5d3a (Credits to Jacob Ferus)
    '''

    # create style folder if not exists
    stylelib_path = f"{mpl.get_configdir()}/stylelib"
    Path(stylelib_path).mkdir(exist_ok=True)
    
    # download the styles from the github-repo if they don't exist
    for style in ["rose-pine-dawn.mplstyle", "rose-pine-moon.mplstyle", "rose-pine.mplstyle"]:
        filename = f"{stylelib_path}/{style}"
        if not overwrite and os.path.isfile(filename):
            continue
        # fetch and add to folder
        content = requests.get(f"https://raw.githubusercontent.com/h4pZ/rose-pine-matplotlib/main/themes/{style}").text
        with open(filename, "w+") as f:
            f.write(content)


def plot_emotion_dist(data, outpath):
    '''
    Plots an overall distribution of the emotions across all headlines in the data.

    Args:
    -   data (pandas.DataFrame): Dataframe with the summary of the classification results.
    -   outpath (pathlib.PosixPath): Path to output data.

    '''

    # add the styles to the style library
    add_rose_pine_styles(overwrite=False)

    # set the style
    plt.style.use("rose-pine-dawn")
    
    # set the color palette
    colors = ['#FF6E78', '#EA9E70', '#F6C177', '#FAD487', '#ABCB8E', '#87DFE4', '#B48EAD']
    sns.set_palette(sns.color_palette(colors))

    # set the font
    mpl.rcParams['font.family'] = 'Times New Roman'

    # create the bar plot
    ax = sns.barplot(data=data, x='predicted_emotion', y='all_headlines')

    # add labels and titles
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Number of headlines')
    ax.set_title('Distribution of emotions across all headlines', fontweight='bold')

    # save the plot to file
    plt.savefig(outpath / "emotion_distribution", dpi=300, bbox_inches='tight')

    
def plot_emotions_by_label(data, outpath):
    '''
    Creates pie charts that show the distribution of emotions across real and fake headlines.

    Args:
    -   data (pandas.DataFrame): Dataframe with the summary of the classification results.
    -   outpath (pathlib.PosixPath): Path to output data.
    '''
    
    # set the color palette
    colors = ['#FF6E78', '#EA9E70', '#F6C177', '#FAD487', '#ABCB8E', '#87DFE4', '#B48EAD']
    sns.set_palette(sns.color_palette(colors))

    # create the subplots for the pie plots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # set general title with good space underneath
    fig.suptitle('Distribution of emotions across real and fake headlines', fontsize=32, fontweight='bold', fontfamily='Times New roman', y=1.025)

    # first do piechart for real headlines
    ax = axes[0]
    ax.pie(data['real_only'], labels=data['predicted_emotion'], autopct='%1.1f%%', startangle=90, labeldistance=1.05, pctdistance=0.75, textprops={'fontsize': 14, 'fontfamily': 'Times New Roman',}, colors=colors, wedgeprops={'linewidth': 0.75, 'edgecolor': 'white'})
    ax.axis('equal')
    ax.set_title('Real headlines', fontsize=24, fontweight='bold', fontfamily='Times New Roman')

    # then we do piechart for fake headlines
    ax = axes[1]
    ax.pie(data['fake_only'], labels=data['predicted_emotion'], autopct='%1.1f%%', startangle=90, labeldistance=1.05, pctdistance=0.75, textprops={'fontsize': 14, 'fontfamily': 'Times New Roman'}, colors=colors, wedgeprops={'linewidth': 0.75, 'edgecolor': 'white'})
    ax.axis('equal')
    ax.set_title('Fake headlines', fontsize=24, fontweight='bold', fontfamily='Times New Roman')

    # set the background color to a light-brown
    fig.set_facecolor('#6E4827')

    # save the plot to file
    plt.savefig(outpath / "emotions_by_label", dpi=300, bbox_inches='tight')


def main():
    # parse arguments
    args = arg_parse()

    # define paths
    inpath, outpath = define_paths(args)

    # load data
    emotion_data = pd.read_csv(inpath)

    # get summary of classification
    clf_summary = get_clf_summary(emotion_data, outpath)

    # plot emotion distribution
    plot_emotion_dist(clf_summary, outpath)

    # plot emotions by label
    plot_emotions_by_label(clf_summary, outpath)

    # print success message
    print("Visualizations complete! They are saved in {}.".format(outpath))


# run main
if __name__ == "__main__":
    main()
