# ML-Final-Project

## Compilation instructions

This project requires the `nltk` package. To install the `nltk` package, run the following code in your terminal.  

```
pip install nltk
```

Final code for this project is compiled in `project_streamline_features.py`.

## Data preprocessing

- Remove geo, media, and author columns because they contain no data. 
- Remove unrelated tweets: CSGO, Covid, “spreading/spread like wildfire”
- Remove non-English tweets
- Remove retweets
-	Removing special characters
-	Extract features: caps, exclamations, hashtag, urls, wordtype

## Model building 
- Labeling data: use NLTK sentiment analyzer
- Train Naive Bayes Classifier and validate using NLTK

## Various ideas we explored (but didn't work)
- Compare tweets via parse tree similarities
- Weak supervision methods
