from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, regexp_extract

spark = SparkSession.builder \
    .appName("Load All Books") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

books_dir = "/Users/bvijay/MLBD_Assignment1/spark/input"

print("="*60)
print("Loading ALL books with Spark...")
print("="*60)

# Let Spark read all .txt files directly
books_df = spark.read.text(books_dir + "/*.txt") \
    .withColumn("full_path", input_file_name()) \
    .withColumn("file_name", regexp_extract("full_path", r'([^/]+\.txt)$', 1)) \
    .select("file_name", "value") \
    .withColumnRenamed("value", "text") \
    .filter("text != ''")

# Group by file to combine lines into full text
from pyspark.sql.functions import collect_list, concat_ws

books_df = books_df.groupBy("file_name") \
    .agg(concat_ws("\n", collect_list("text")).alias("text"))

total_books = books_df.count()
print("Total books loaded: {}".format(total_books))

books_df.select("file_name").show(truncate=False)

# Save
books_df.write.parquet("books_df.parquet", mode="overwrite")

print("="*60)
print("Saved {} books to: books_df.parquet".format(total_books))
print("="*60)

spark.stop()

