This project aims to classify news articles as real or fake using Natural Language Processing (NLP) and a deep learning model (RNN and LSTM). 
used libraries: numpy, pandas, nltk, gensim, matplotlib, wordcloud, plotly, tensorflow, scikit-learn, seaborn

The dataset includes two CSV files, True.csv and Fake.csv, which are combined, cleaned, and processed for training and evaluation.
Datasets are provided in True.zip and Fake.zip, each containing True.csv and Fake.csv.

Data Preprocessing: combine title and text columns into a single text column, clean text by removing stop words, short tokens, and unwanted characters.

Visualization: generate word clouds to analyze common terms in real and fake news, plot histograms of article lengths.

Model Training: tokenize and pad text data to prepare it for deep learning, train a Bidirectional LSTM model with embedding and dense layers to classify news articles.

Evaluation: evaluate the model's performance using accuracy and a confusion matrix.

Test the model with a custom news article.
