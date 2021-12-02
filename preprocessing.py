from pyspark.sql.functions import collect_list, array_join, udf
from pyspark.sql.types import ArrayType, StringType
from pyspark import SparkConf
from pyspark.sql import SparkSession
import re
import nltk
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import jaccard_score
# from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def lemmitizeWords(text):
    # token = TokTokTokenizer()
    # words=token.tokenize(text)
    lemma = WordNetLemmatizer()
    words = nltk.tokenize.word_tokenize(text)
    listLemma=[]
    for w in words:
        x=lemma.lemmatize(w, pos="v")
        listLemma.append(x)
    return ' '.join(map(str, listLemma))

def stopWordsRemove(text):
    from nltk.corpus import stopwords
    words = nltk.tokenize.word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered = [w for w in words if not w in stop_words]
    return ' '.join(map(str, filtered))



def strip_list_noempty(mylist):
    newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)
    return [item for item in newlist if item != '']



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
all_tags = [item for sublist in all_tags for item in sublist]
# all_tags_tokenized = [nltk.tokenize.word_tokenize(i) for i in all_tags]
# tagDist = nltk.FreqDist(all_tags_tokenized)
tagDist = nltk.FreqDist(all_tags)
tagDist2 = nltk.FreqDist(tagDist)
frequencies = tagDist2.most_common(500)
tag_features = [tag[0] for tag in frequencies]

def most_common(tags):
    tags_filtered = []
    for i in range(0, len(tags)):
        if tags[i] in tag_features:
            tags_filtered.append(tags[i])
    return tags_filtered

def clean_punct(text):
    # token = TokTokTokenizer()
    # words=token.tokenize(text)
    punct = '!"#$%&\'()*+,./:;<=>?@[\]^_`{|}~'
    words = nltk.tokenize.word_tokenize(text)
    punctuation_filtered = []
    regex = re.compile('[%s]' % re.escape(punct))
    remove_punctuation = str.maketrans(' ', ' ', punct)
    for w in words:
        if w in tag_features:
            punctuation_filtered.append(w)
        else:
            punctuation_filtered.append(regex.sub('', w))
  
    filtered_list = strip_list_noempty(punctuation_filtered)
        
    return ' '.join(map(str, filtered_list))

#Remove all questions that don't have one of the tags in the feature
# cleaned_df['Tag'] = cleaned_df.select("tag").apply(lambda x: most_common(x))
# cleaned_df = cleaned_df.rdd.map(lambda x: most_common(x['Tag']))
most_common_udf = udf(most_common, ArrayType(StringType()))
cleaned_df = cleaned_df.withColumn("Tag", most_common_udf(cleaned_df.Tag))
length_zero_udf = udf(lambda x: x if len(x)>0 else None, ArrayType(StringType()))
# cleaned_df['Tag'] = cleaned_df['Tag'].apply(lambda x: x if len(x)>0 else None)
cleaned_df = cleaned_df.withColumn("Tag", length_zero_udf(cleaned_df.Tag))
cleaned_df = cleaned_df.dropna()

#Then the body needs cleaned up
#This includes removing the HTML, removing punctuation, lemmatizing words and removing stop words
bsUDF = udf(lambda x: BeautifulSoup(x).get_text())
cleaned_df = cleaned_df.withColumn("Body", bsUDF(cleaned_df.Body))
clean_text_udf = udf(lambda x: clean_text(x))
cleaned_df = cleaned_df.withColumn("Body", clean_text_udf(cleaned_df.Body))
clean_punct_udf = udf(lambda x: clean_punct(x))
cleaned_df = cleaned_df.withColumn("Body", clean_punct_udf(cleaned_df.Body))
lemmatizeWords_udf = udf(lambda x: lemmitizeWords(x))
cleaned_df = cleaned_df.withColumn("Body", lemmatizeWords_udf(cleaned_df.Body))
stopWordsRemove_udf = udf(lambda x: stopWordsRemove(x))
cleaned_df = cleaned_df.withColumn("Body", stopWordsRemove_udf(cleaned_df.Body))
#Then the titles need cleaned up
cleaned_df = cleaned_df.withColumn("Title", bsUDF(cleaned_df.Title))
cleaned_df = cleaned_df.withColumn("Title", clean_text_udf(cleaned_df.Title))
cleaned_df = cleaned_df.withColumn("Title", clean_punct_udf(cleaned_df.Title))
cleaned_df = cleaned_df.withColumn("Title", lemmatizeWords_udf(cleaned_df.Title))
cleaned_df = cleaned_df.withColumn("Title", stopWordsRemove_udf(cleaned_df.Title))

#Then we need to apply our models
# cleaned_df = cleaned_df.withColumn("Tag", array_join("Tag", ","))

# cleaned_df.write.csv("hdfs://columbia:30141/output/test")

cleaned_df_pd = cleaned_df.toPandas()

x1 = cleaned_df_pd["Body"]
x2 = cleaned_df_pd["Title"]
y1 = cleaned_df_pd["Tag"]

mlb = MultiLabelBinarizer()

y_bin = mlb.fit_transform(y1)

vectorizer_X1 = TfidfVectorizer(analyzer = 'word',
    min_df=0.0,
    max_df = 1.0,
    strip_accents = None,
    encoding = 'utf-8', 
    preprocessor=None,
    token_pattern=r"(?u)\S\S+",
    max_features=1000)

vectorizer_X2 = TfidfVectorizer(analyzer = 'word',
    min_df=0.0,
    max_df = 1.0,
    strip_accents = None,
    encoding = 'utf-8', 
    preprocessor=None,
    token_pattern=r"(?u)\S\S+",
    max_features=1000)


X1_tfidf = vectorizer_X1.fit_transform(x1)
X2_tfidf = vectorizer_X2.fit_transform(x2)
X_tfidf = hstack([X1_tfidf,X2_tfidf])

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_bin, test_size = 0.2, random_state = 42)

mn = MultinomialNB()
clf = OneVsRestClassifier(mn)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

jaccard_score(y_test, y_pred, average=None)

jacc_list = [jaccard_score]

jacc_rdd = sc.parallelize(jacc_list)

jacc_rdd.coalesce(1).saveAsTextFile("hdfs://columbia:30141/output/test/")
