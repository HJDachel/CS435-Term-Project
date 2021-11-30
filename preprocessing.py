from pyspark.sql.functions import collect_list
from pyspark import SparkConf
from pyspark.sql import SparkSession

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

joined_df.write.csv("hdfs://columbia:30141/output/test.csv")