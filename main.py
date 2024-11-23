import numpy as np
import pandas as pd
#load the data into dataFrames
#dataFrames are 2 dimensional data structure

df_true = pd.read_csv("True.csv")
df_fake = pd.read_csv("Fake.csv")

#check if there are null values in the dataFrame
#print(df_true.isnull().sum())
#print(df_fake.isnull().sum())

df_true["isFake"] = 0
df_fake["isFake"] = 1
#print(df_true.head(10))
#print(df_fake.head(10))

#concatenate the 2 df into one df, updating the old indexes into a consecutive value
df= pd.concat([df_true,df_fake]).reset_index(drop=True)
#print(df.columns)

#drop the date column because it's not useful
#inplace is used to change the dataframe in the memory as well
df.drop(columns = ["date"], inplace = True)
#print(df.columns)

#combine the title and text columns
df["original"] = df["title"] + " "  + df["text"]
#print(df.columns)


## --> DATA CLEANING
import nltk
import gensim

nltk.download('stopwords')
nltk.download('punkt_tab')


#create a list with not so important words
from nltk.corpus import stopwords, words

stop_words = stopwords.words('english')
stop_words.extend(["from", "subject", "re", "edu", "use"])

#Remove stop words from our df and words with 2 or fewer letters
#gendim.utils.simple_preprocess converts the text into lowercase,
#splits the text into a list of strings and removes tokens shorter than 2 characters

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2 and token not in stop_words:
            result.append(token)
    return result

df["clean"] = df["original"].apply(preprocess)

#print(df["original"][0])
#print(df["clean"][0])

#obtain the total number of words in the dataset
list_of_words = []
for i in df.clean:
    for j in i:
        list_of_words.append(j)

#print(list_of_words)
#print(len(list_of_words))

#obtain total number of unique words
total_words = len(set(list_of_words))
#print(total_words)

## QUESTION:de ce am mai multe cuvinte decat cele din curs?

#create a new column with a massive string of all the cleaned words
df['clean_joined'] = df['clean'].apply(lambda x: ' '.join(x))

#DATA VISUALIZATION

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

#plot the word cloud for text that is real news
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = stop_words).generate(" ".join(df[df.isFake == 1].clean_joined))
plt.imshow(wc, interpolation = 'bilinear')
plt.axis('off')
plt.show()

#and the same for fake news
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = stop_words).generate(" ".join(df[df.isFake == 0].clean_joined))
plt.imshow(wc, interpolation = 'bilinear')
plt.axis('off')
plt.show()

#maximum length for word embeddings

maxlen=-1
for doc in df.clean_joined:
    tokens = nltk.word_tokenize(doc)
    if(len(tokens) > maxlen):
        maxlen = len(tokens)
print("The maximum number of words in any document is ", maxlen)

import plotly.express as px

fig = px.histogram(x = [len(nltk.word_tokenize(i)) for i in df.clean_joined ], nbins= 100)
fig.show()

import tensorflow as tf
#TOKENIZARE
from sklearn.model_selection import train_test_split
#80% data is for training, 20% is for testing

x_train, x_test, y_train, y_test = train_test_split(df.clean_joined, df.isFake, test_size = 0.2)

from nltk import word_tokenize
# Incearca sa limitezi numarul de cuvinte la 10.000 (sau altă valoare rezonabilă)
# Dimensiunea vocabularului este acum limitată
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(x_train)
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

# Padding secvențelor pentru a le uniformiza
maxlen = 40
padded_train = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=maxlen, padding='post', truncating='post')
padded_test = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=maxlen, truncating='post')

# Definirea modelului
# Define the model
model = tf.keras.models.Sequential()

# Add Embedding layer
model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=128))

# Add Bidirectional LSTM layer
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)))

# Add Dense layers
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Now, model.build() will force the model to initialize correctly based on the input dimensions
model.build(input_shape=(None, 40))  # 40 is the length of the padded sequences

# Summary of the model
model.summary()

y_train = np.asarray(y_train)
model.fit(padded_train, y_train, batch_size=64, validation_split=0.1, epochs=2)

pred = model.predict(padded_test)
prediction = []
for i in range(len(pred)):
    if pred[i].item() > 0.5:
        prediction.append(1)
    else:
        prediction.append(0)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(list(y_test), prediction)
print("Model Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(list(y_test), prediction)
plt.figure(figsize = (20,20))
import seaborn as sns
sns.heatmap(cm, annot=True)
plt.show()

###test
print("Testareea unei stiri")
# Textul știrii
news_text = "New York prosecutors have vowed to oppose President-elect Donald Trump’s effort to overturn his criminal conviction, but they expressed a willingness to wait to sentence him until he completes his upcoming presidential term. Prosecutors urged the judge who is overseeing Trump's sentence after his guilty verdict to consider options other than dismissal, including holding off until the president-elect is scheduled to leave the White House in 2029. The Manhattan district attorney's office asked Justice Juan Merchan to set a new deadline of 9 December for both sides to consider the case and file new motions.Trump's sentencing is scheduled for 26 November, but it could be delayed further."
# Aplica preprocesarea
clean_news = preprocess(news_text)

# Transformă textul curățat într-o singură secvență de cuvinte
clean_news_joined = ' '.join(clean_news)

# Tokenizare
news_sequence = tokenizer.texts_to_sequences([clean_news_joined])

# Padding
padded_news = tf.keras.preprocessing.sequence.pad_sequences(news_sequence, maxlen=40, padding='post', truncating='post')

# Predicția
pred = model.predict(padded_news)

# Verifică dacă predicția este mai mare de 0.5 pentru a o considera "falsă"
if pred[0].item() > 0.5:
    print("Știrea este probabil falsă.")
else:
    print("Știrea este probabil reală.")

