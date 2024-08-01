import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.cluster import AgglomerativeClustering
import re
import spacy
from nltk.corpus import stopwords


data = pd.read_excel('E:\\table\\EMR_data\\SIS_EMR\\patient_information1.xlsx')
data.dropna(subset=['Procedure'], inplace=True)
#drop
data = data[~data['Procedure'].isin(['none', 'Unknown'])]
# 1.1 data clean
def clean_text(text):
    text = re.sub('<.*?>', '', text)  # 去除HTML标签
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # 去除特殊字符和数字，只保留字母和空格
    return text

data['Procedure'] = data['Procedure'].apply(clean_text)

# 1.2 token
nlp = spacy.load('en_core_web_sm')
def tokenize_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens

data['Procedure'] = data['Procedure'].apply(tokenize_text)

# 1.3 去除停用词
stop_words = set(stopwords.words('english'))
def remove_stopwords(tokens):
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

data['Procedure'] = data['Procedure'].apply(remove_stopwords)
# 1.4 小写化
data['Procedure'] = data['Procedure'].apply(lambda x: [word.lower() for word in x])
# 1.5 去除空白词
data['Procedure'] = data['Procedure'].apply(lambda x: [word for word in x if word.strip()])
surgeries = data['Procedure']

# 2. text vector
# Create TaggedDocument objects for Doc2Vec model
tagged_data = [TaggedDocument(words=words, tags=[str(i)]) for i, words in enumerate(data['Procedure'])]

# Train a Doc2Vec model
doc2vec_model = Doc2Vec(vector_size=300, window=2, min_count=1, workers=4, epochs=100)
doc2vec_model.build_vocab(tagged_data)
doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

# Get vectors for all the descriptions
vectors = [doc2vec_model.dv[i] for i in range(len(data))]

# Use Agglomerative Clustering to cluster the descriptions
agg_cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=15, linkage='ward')
clusters = agg_cluster.fit_predict(vectors)

# Add the cluster labels to your DataFrame
data['Cluster'] = clusters

# Save the clustered data to a new Excel file
data.to_excel('E:\\table\\EMR_data\\SIS_EMR\\output_results_threshold_15.xlsx', index=False)

