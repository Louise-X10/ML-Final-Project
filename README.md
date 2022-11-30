# ML-Final-Project

Place to upload code and collaborate. 

## Data preprocessing

- Remove geo, media, and author columns because they contain no data. 
- Remove unrelated tweets: CSGO, Covid, “spreading/spread like wildfire”
- Remove non-English tweets
- Remove retweets and log frequency of retweets
-	Removing special characters
-	Extract features: caps, exclamations, hashtag, key words (Adrian), links (“news” in expanded urls), frequency of retweets
-	Compare parse trees of tweets: structural kernel

## Model building 
-	Training the model: clustering model
-	Validation: label some data and see which cluster they end up in (semi-supervised learning), pick some data from each cluster and trace back to see what they look like
-	Present results (graph of the clusters): use t-SNE to reduce graph dimensionality
