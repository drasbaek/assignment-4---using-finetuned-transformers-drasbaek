# Assignment 4: Text Classification using Finetuned Transformers

## Repository Overview
1. [Description](#description)
2. [Repository Tree](#tree)
3. [Usage](#gusage)
4. [Modified Usage](#musage)
5. [Results](#results)
6. [Discussion](#discussion)


## Description <a name="description"></a>
This repository includes the solution by *Anton Drasbæk Schiønning (202008161)* to assignment 4 in the course "Language Analytics" at Aarhus University.

It provides a framework for doing emotion classification of headlines from the [Fake News Dataset](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news) by utilizing a Huggingface pipeline. The dataset consists of over 7000 news headlines, texts and corresponding labels (real/fake). The HuggingFace model used to do the classification is [*j-hartmann/emotion-english-distilroberta-base*](j-hartmann/emotion-english-distilroberta-base) which is a fine-tuned version of the destilled RoBERTa model for emotion classification. <br>

Visualizations are also made to provide an overview of the classifications.
<br/><br/>

## Repository Tree <a name="tree"></a>

```
.
├── README.md
├── assign_desc.md                                                  
├── data
│   ├── classified_titles_emotion-english-distilroberta-base.csv   <---- headlines with classifications and score
│   └── fake_or_real_news.csv                                      <---- original dataset for real/fake news
├── out
│   └── results_emotion-english-distilroberta-base    <---- example results
│       ├── classification_overview.csv                   <---- overview of all classifications         
│       ├── emotion_distribution.png                      <---- distribution of all emotions
│       └── emotions_by_label.png                         <---- share of emotions by headline type
├── requirements.txt
├── run.sh
├── setup.sh
└── src
    ├── classify.py                                   <---- script for running classifications                                         
    └── visualize.py                                  <---- script for creating visualizations/outputs
```
<br>

## Usage <a name="gusage"></a>
This analysis only assumes that you have Python3 installed and clone this GitHub repository. When this has been done, you can run the full analysis with the shell script:
```
bash run.sh
```

This will achieve the following:
* Create and activate a virtual environment
* Install requirements to that environment
* Classify emotions in all headlines (`classify.py`) using *j-hartmann/emotion-english-distilroberta-base*
* Create and save visualizations of the classifications (`visualize.py`)
* Deactivate the environment
<br>

The results are saved to the `out` directory under a subfolder, named after the model used. This result contains three files:
* `classification_overview.csv`: Csv file with overview of how many headlines were classified as each emotion. Also splits the classifications by real and fake headlines.
* `emotion_distribution.png`: Bar chart showing the distribution of emotions identified across all headlines.
* `emotion_by_label.png`: Pie charts showing the distribution of emotions for real and fake headlines, presented side-by-side for an easy comparison.

Examples of these three files are also seen under [Results](#results).
<br/><br/>

## Modified Usage <a name="musage"></a>
If you wish to use a different model for the emotion classifications, the repository also allows running a modified analysis. Firstly, run the setup bash script to create an environment and install requirements:
```
bash setup.sh
```

### Run classifications
By default, `classify.py` uses *j-hartmann/emotion-english-distilroberta-base* for classifications. However, any other pretrained model from [Huggingface](https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads) for text classification can be used. Please note, that you should select a model specific to emotion classification if you wish to maintain the scope of the analysis. 

When having selected a model, run classifications as such:
```
# uses distilbert-base-uncased-go-emotions-student for classification
python src/classify.py -m "joeddav/distilbert-base-uncased-go-emotions-student"
```
You can find the data with classifications in `data/classified_titles_{SELECTED_MODEL_NAME}`.
<br/><br/>

### Create Visualizations
Visualizations for a classification file can be created done by running the `visualize.py` file. Again, you must specify which model was used for classifying the data in order for the visualization to cover the right data file:
```
python src/visualize.py -m "joeddav/distilbert-base-uncased-go-emotions-student"
```
From this, you will get a folder named `out/results_{SELECTED_MODEL_NAME}` which contains the three files mentioned earlier. <br>

**PLEASE NOTE**: *Visualizations are made to look neatly for classifications that use 7 emotions. If there are more or fewer in your model, visualizations may not look as neat. Regardless, `classification_overview.csv` for your classifications should still provide the needed overview.* 
<br/><br/>

## Results <a name="results"></a>
Below are the results for running the classification with the default model, which can be found in the directory `out/results_emotion-english-distilroberta-base`.

### Table: Classifcation Overview
| Predicted Emotion | All Headlines | Real Only | Fake Only |
|------------------ | ------------- | --------- | --------- |
| Anger             | 795           | 383       | 412       |
| Disgust           | 434           | 186       | 248       |
| Fear              | 1076          | 555       | 521       |
| Joy               | 155           | 63        | 92        |
| Neural            | 3180          | 1649      | 1531      |
| Sadness           | 487           | 245       | 242       |
<br>

### Plot: Emotion Distribution
![alt text](https://github.com/AU-CDS/assignment-4---using-finetuned-transformers-drasbaek/blob/main/out/results_emotion-english-distilroberta-base/emotion_distribution.png?raw=True)
<br>

### Plot: Emotions by Label
![alt text](https://github.com/AU-CDS/assignment-4---using-finetuned-transformers-drasbaek/blob/main/out/results_emotion-english-distilroberta-base/emotions_by_label.png?raw=True)
<br>

## Discussion <a name="discussion"></a>
Overall, the pie charts above reveal that the distribution of emotions in headlines is strikingly similar across the real and fake news. For both label types, the most common emotion by far is *neutral* with a 52% share for real headlines and 48% for fake ones. Also, *joy* is the rarest emotion in both the real and fake headlines. Perhaps the most noticeable discrepancy is that 7.8% of fake headlines are classified as *disgust* whereas it is just 5.9% for real ones. <br>

The main takeaway remains that the fake headlines are extremely similar to the real ones, when it comes to the primary emotion displayed, according to this analysis. This implies that emotions in the headlines are not a good indicator of whether or not it is a real headline. Still, it should be emphasized that these results are just based on the classfications by *j-hartmann/emotion-english-distilroberta-base* and using a different classification model may have produced different results. If interested, exploring this can easily be achieved by following [Modified Usage](#musage).



