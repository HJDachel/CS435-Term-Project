from pyspark.sql.functions import collect_list, array_join
from pyspark import SparkConf
from pyspark.sql import SparkSession
import re
import nltk
import bs4 as bs

def lemmitizeWords(text):
    token = TokTokTokenizer()
    words=token.tokenize(text)
    listLemma=[]
    for w in words:
        x=lemma.lemmatize(w, pos="v")
        listLemma.append(x)
    return ' '.join(map(str, listLemma))

def stopWordsRemove(text):
    token = TokTokTokenizer()
    stop_words = set(stopwords.words("english"))
    words=token.tokenize(text)
    filtered = [w for w in words if not w in stop_words]
    return ' '.join(map(str, filtered))

def clean_punct(text):
    token = TokTokTokenizer()
    words=token.tokenize(text)
    punctuation_filtered = []
    regex = re.compile('[%s]' % re.escape(punct))
    remove_punctuation = str.maketrans(' ', ' ', punct)
    for w in words:
        if w in tags_features:
            punctuation_filtered.append(w)
        else:
            punctuation_filtered.append(regex.sub('', w))
  
    filtered_list = strip_list_noempty(punctuation_filtered)
        
    return ' '.join(map(str, filtered_list))

def strip_list_noempty(mylist):
    newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)
    return [item for item in newlist if item != '']

def most_common(tags, tags_features):
    tags_filtered = []
    for i in range(0, len(tags)):
        if tags[i] in tags_features:
            tags_filtered.append(tags[i])
    return tags_filtered

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'\n", " ", text)
    text = re.sub(r"\'\xa0", " ", text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text



conf = SparkConf()
conf.setMaster("spark://columbia:30160")
conf.setAppName("preprocessing")
conf.set("spark.executor.memory", "1g")

spark = SparkSession.builder \
     .master("spark://columbia:30160") \
     .appName("preprocessing") \
     .getOrCreate()
sc = spark.sparkContext


questions_df = spark.read.option("header",True).csv("hdfs://columbia:30141/input/Questions.csv")

tags_df = spark.read.option("header",True).csv("hdfs://columbia:30141/input/Tags.csv")

tags_grouped_by_id = tags_df.groupby('Id').agg(collect_list('Tag').alias("Tag"))

joined_df = questions_df.join(tags_grouped_by_id, 'Id')

#Convert Tag column from array to string
# joined_flattened_df = joined_df.withColumn("Tag", array_join("Tag", ","))

joined_df_filtered = joined_df.filter(joined_df.Score > 0)

#Maybe we want to drop score as well because we won't be using it at all
cleaned_df = joined_df_filtered.drop('CreationDate', 'OwnerUserId', 'Id','ClosedDate')

#Tags gets cleaned up
# cleaned_df['Tag'] = cleaned_df['Tag'].apply(lambda x : x.split())
# all_tags = tags_df.rdd.map(lambda x: x[1]).collect()
all_tags = cleaned_df.rdd.map(lambda x: x[3]).collect()
#Create the feature vector
all_tags_tokenized = [nltk.tokenize.word_tokenize(i) for i in all_tags]
tagDist = nltk.FreqDist(all_tags_tokenized)
tagDist2 = nltk.FreqDist(tagDist)
frequencies = tagDist2.most_common(500)
tag_features = [tag[0] for tag in frequencies]
#Remove all questions that don't have one of the tags in the feature
cleaned_df['Tag'] = cleaned_df['Tag'].apply(lambda x: most_common(x))
cleaned_df['Tag'] = cleaned_df['Tag'].apply(lambda x: x if len(x)>0 else None)
cleaned_df.dropna(subset=['Tag'], inplace=True)

#Then the body needs cleaned up
#This includes removing the HTML, removing punctuation, lemmatizing words and removing stop words
cleaned_df['Body'] = cleaned_df['Body'].apply(lambda x: BeautifulSoup(x).get_text())
cleaned_df['Body'] = cleaned_df['Body'].apply(lambda x: clean_text(x)))
cleaned_df['Body'] = cleaned_df['Body'].apply(lambda x: clean_punct(x)))
cleaned_df['Body'] = cleaned_df['Body'].apply(lambda x: lemmitizeWords(x)))
cleaned_df['Body'] = cleaned_df['Body'].apply(lambda x: stopWordsRemove(x)))
#Then the titles need cleaned up
cleaned_df['Title'] = cleaned_df['Body'].apply(lambda x: BeautifulSoup(x).get_text())
cleaned_df['Title'] = cleaned_df['Body'].apply(lambda x: clean_text(x))
cleaned_df['Title'] = cleaned_df['Body'].apply(lambda x: clean_punct(x))
cleaned_df['Title'] = cleaned_df['Body'].apply(lambda x: lemmitizeWords(x))
cleaned_df['Title'] = cleaned_df['Body'].apply(lambda x: stopWordsRemove(x))

#Then we need to apply our models

cleaned_df.write.csv("hdfs://columbia:30141/output/test")