# Data Mining Project

# preprocess

import pandas as pd
import os
import numpy as np

path="./book_info"
file_list=os.listdir(path)

csv_base='./book_info/{}'
csv=csv_base.format(file_list[0])
df_total=pd.read_csv(csv,encoding='utf-8')

for i in range(1,len(file_list)):
    csv=csv_base.format(file_list[i])
    df=pd.read_csv(csv,encoding='utf-8')
    df_total=pd.concat([df_total,df],axis=0).reset_index(drop=True)

# Preprocess 'Title'

df_total=df_total.dropna(subset=['title'])

# Preprocess duplicates

df_total=df_total.drop_duplicates(['title'])
df_total=df_total.reset_index(drop=True)

# Preprocess nulls in 'introduction' and 'review' column

df_total=df_total.dropna(subset=['introduction','review'],how='all')

df_total=df_total.fillna('')
df_total=df_total.reset_index(drop=True)

# Preprocess as 'introduction'+'review'='text'

df_total['text']=df_total[['introduction','review']].apply(lambda x:' '.join(x),axis=1)

# Save as .csv

df_total.to_csv('./book_info(total).csv',index=False,encoding='utf-8-sig')

# Visualiztion and Statistics

import pandas as pd
from matplotlib import pyplot as plt

df=pd.read_csv('./book_info(total).csv',encoding='utf-8')

len_text=[len(df.text[i]) for i in range(len(df))]

avg_len=0
n=0
for i in range(len(df)):
    if len(df.text[i])<10000:
        avg_len+=len(df.text[i])
        n=n+1
print(avg_len/n)
plt.hist(len_text,50)
plt.show()

# TF-IDF Model

import gc
import konlpy
from konlpy.tag import Kkma,Okt,Komoran
from pprint import pprint
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

book=pd.read_csv('./book_info(total).csv',encoding='utf-8')
book.loc[len(book)]=['','','',0.0,'','','','','','{}'.format(input())]

okt=Okt()
total_books=[]
book_content=book['text']

for books in book_content:
    pos_books=[','.join(t[:-1]) for t in okt.pos(books) if (t[1]=='Noun')]
    total_books.append(' '.join(pos_books))
    
kor_vectorizer = CountVectorizer()
kor_bow = kor_vectorizer.fit_transform(total_books)

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(kor_bow.toarray())

def find_most_similar_book_idf(index, tfidf, book):
    idx = (-cosine_similarity(tfidf[index], tfidf)[0]).argsort()[1:6]
    return idx

idx=len(book)-1
idx=find_most_similar_book_idf(idx,tfidf,book)

for i in range(5):
    print(book.title[idx[i]])
    
# Word2Vec Model

import pandas as pd
import nltk
from konlpy.tag import Okt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

df=pd.read_csv('./book_info(total).csv')

# Stopwords
df_stop=pd.read_csv('./한국어불용어100.txt',sep='\t',header=None,
                   names=['stopword','else1','else2'])
stop_word=[df_stop['stopword'][i] for i in range(len(df_stop))]

okt=Okt()
x_train=[]
for sentence in df['text']:
    temp_x=[','.join(t[:-1]) for t in okt.pos(sentence) if ((t[1]=='Noun') & (t[0] not in stop_word))]
    x_train.append(temp_x)
    
from gensim.models import Word2Vec

model=Word2Vec(x_train,size=150,window=10,min_count=100,workers=4)

word_vectors=model.wv
vocabs=word_vectors.vocab.keys()
word_vectors_list=[word_vectors[v] for v in vocabs]

# Visualization and Statistics

list=[word for word in vocabs]

count_list=[]
for j in range(len(list)):
    count=0
    for i in range(len(df)):
        count+=df.text[i].count(list[j])
    count_list.append(count)

count_list=np.array(count_list)
s=count_list.argsort()
top_ten_count=np.array([count_list[s[len(s)-i-300]] for i in range(11)])

plt.rc('font',family='Malgun Gothic')
plt.scatter(['벽','단어','활용','투자','자기','이의','기출','학교','인지','사용','사실'],top_ten_count)

plt.rc('font', family='Malgun Gothic')

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

model2=TSNE(learning_rate=100)
transformed=model2.fit_transform(word_vectors_list)

xs2=transformed[:,0]
ys2=transformed[:,1]

plt.figure(figsize=(28,21))

plt.scatter(xs2,ys2)
for i,v in enumerate(vocabs):
    plt.annotate(v,xy=(xs2[i],ys2[i]))
    
plt.show()

# Word2Vec model example

return_doc=model.wv.most_similar(positive=['일본','추리'])

for i in range(5):
    print(return_doc[i])

# Document Vectorize and Model    
    
model.init_sims(replace=True)

def document_vector(model,doc):
    doc=doc.split(' ')
    doc=[doc[i] for i in range(len(doc)) if doc[i] in vocabs]
    mean=0
    for i in range(len(doc)):
        mean+=model.wv[doc[i]]/len(doc)
    return mean

import math

def cos_sim(a,b):
    s=0
    s1=0
    s2=0
    for i in range(len(a)):
        s=s+a[i]*b[i]
        s1=s1+a[i]**2
        s2=s2+b[i]**2
    return s/math.sqrt(s1*s2)

def find_similarist_doc(model,doc):
    a=document_vector(model,doc)
    cos_vec=np.array([0.0]*len(df))
    for i in range(len(df)):
        b=document_vector(model,df.text[i])
        try:
            cos_vec[i]=-cos_sim(a,b)
        except:
            cos_vec[i]=0
    s=cos_vec.argsort()
    return (s[0],s[1],s[2],s[3],s[4])

doc='히가시노 게이고의 살인 관련 추리 소설'

s=find_similarist_doc(model,doc)

for i in range(5):
    print(df.title[s[i]])
    
# Doc2Vec Model
    
def tokenizer_okt_nouns(doc):
    return okt.nouns(doc)
df['token_text']=df['text'].apply(tokenizer_okt_nouns)

doc_df=df[['title','token_text']].values.tolist()
tagged_data=[TaggedDocument(words=_d,tags=[uid]) for uid,_d in doc_df]

max_epochs=10

model3=Doc2Vec(window=10,vector_size=150,alpha=0.025,min_alpha=0.025,min_count=100,dm=1,negative=5,seed=9999)

model3.build_vocab(tagged_data)

for epoch in range(max_epochs):
    model3.train(tagged_data,total_examples=model.corpus_count,epochs=model.epochs)
    model3.alpha-=0.002
    model3.min_alpha=model.alpha
    
doc_list='히가시노 게이고의 살인 관련 추리 소설'.split(' ')

inferred_vector=model3.infer_vector(doc_list)
return_docs=model3.docvecs.most_similar(positive=[inferred_vector],topn=5)

for i in range(5):
    print(return_docs[i])
