from pyspark.sql.functions import collect_list, array_join
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

#Convert Tag column from array to string
joined_flattened_df = joined_df.withColumn("Tag", array_join("Tag", ","))

joined_df_filtered = joined_flattened_df.filter(joined_df.Score > 0)

cleaned_df = joined_df_filtered.drop('CreationDate', 'OwnerUserId', 'Id','ClosedDate')

cleaned_df.write.csv("hdfs://columbia:30141/output/test")