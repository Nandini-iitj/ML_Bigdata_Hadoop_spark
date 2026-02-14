from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# Initialize Spark with optimized settings
spark = SparkSession.builder \
    .appName("Q10 Book Metadata Extraction") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 70)
print("Question 10: Book Metadata Extraction and Analysis")
print("=" * 70)

# ======================================================================
# STEP 1: LOAD AND EXTRACT METADATA (SINGLE PASS)
# ======================================================================

print("\nLoading and extracting metadata...")

metadata_df = spark.read.parquet("books_df.parquet") \
    .withColumn("title", 
        F.regexp_extract(F.col("text"), r'Title:\s*(.+?)(?:\r?\n)', 1)) \
    .withColumn("author", 
        F.regexp_extract(F.col("text"), r'Author:\s*(.+?)(?:\r?\n)', 1)) \
    .withColumn("release_date", 
        F.regexp_extract(F.col("text"), r'Release Date:\s*(.+?)(?:\r?\n|\[)', 1)) \
    .withColumn("language", 
        F.regexp_extract(F.col("text"), r'Language:\s*(.+?)(?:\r?\n)', 1)) \
    .withColumn("year_str", 
        F.regexp_extract(F.col("release_date"), r'(\d{4})', 1)) \
    .withColumn("year", 
        F.when(F.col("year_str") != "", F.col("year_str").cast(IntegerType()))
         .otherwise(None)) \
    .withColumn("encoding", 
        F.regexp_extract(F.col("text"), r'Character set encoding:\s*(.+?)(?:\r?\n)', 1)) \
    .withColumn("title", F.when(F.col("title") == "", None).otherwise(F.col("title"))) \
    .withColumn("author", F.when(F.col("author") == "", None).otherwise(F.col("author"))) \
    .withColumn("language", F.when(F.col("language") == "", None).otherwise(F.col("language"))) \
    .withColumn("encoding", F.when(F.col("encoding") == "", None).otherwise(F.col("encoding"))) \
    .select("file_name", "title", "author", "year", "language", "encoding", "release_date") \
    .cache()

# Trigger cache with count
total_books = metadata_df.count()
print("Total books loaded: {}".format(total_books))

print("\nSample Metadata:")
metadata_df.show(10, truncate=45)

# ======================================================================
# STEP 2: ALL ANALYSES IN PARALLEL (SINGLE PASS)
# ======================================================================

print("\n" + "=" * 70)
print("RUNNING ANALYSES")
print("=" * 70)

# Compute ALL aggregations in ONE PASS using agg()
stats = metadata_df.agg(
    F.count("*").alias("total_books"),
    F.count(F.when(F.col("title").isNotNull(), 1)).alias("with_title"),
    F.count(F.when(F.col("author").isNotNull(), 1)).alias("with_author"),
    F.count(F.when(F.col("year").isNotNull(), 1)).alias("with_year"),
    F.count(F.when(F.col("language").isNotNull(), 1)).alias("with_language"),
    F.min("year").alias("min_year"),
    F.max("year").alias("max_year")
).collect()[0]

# Books per year
books_per_year = metadata_df \
    .filter(F.col("year").isNotNull()) \
    .groupBy("year") \
    .count() \
    .orderBy("year") \
    .cache()

# Language distribution
language_dist = metadata_df \
    .filter(F.col("language").isNotNull()) \
    .groupBy("language") \
    .count() \
    .orderBy(F.desc("count")) \
    .cache()

# Top authors
top_authors = metadata_df \
    .filter(F.col("author").isNotNull()) \
    .groupBy("author") \
    .count() \
    .orderBy(F.desc("count")) \
    .limit(10) \
    .cache()

# Encoding distribution
encoding_dist = metadata_df \
    .filter(F.col("encoding").isNotNull()) \
    .groupBy("encoding") \
    .count() \
    .orderBy(F.desc("count")) \
    .cache()

# ======================================================================
# STEP 3: DISPLAY RESULTS
# ======================================================================

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print("Total books: {}".format(stats["total_books"]))
print("Books with title: {} ({:.1f}%)".format(
    stats["with_title"], 100 * stats["with_title"] / stats["total_books"]))
print("Books with author: {} ({:.1f}%)".format(
    stats["with_author"], 100 * stats["with_author"] / stats["total_books"]))
print("Books with year: {} ({:.1f}%)".format(
    stats["with_year"], 100 * stats["with_year"] / stats["total_books"]))
print("Books with language: {} ({:.1f}%)".format(
    stats["with_language"], 100 * stats["with_language"] / stats["total_books"]))

if stats["min_year"]:
    print("\nYear range: {} - {}".format(stats["min_year"], stats["max_year"]))

print("\n" + "=" * 70)
print("BOOKS PER YEAR (Top 20)")
print("=" * 70)
books_per_year.show(20)

print("\n" + "=" * 70)
print("LANGUAGE DISTRIBUTION")
print("=" * 70)
language_dist.show(20, truncate=False)

print("\n" + "=" * 70)
print("TOP 10 AUTHORS")
print("=" * 70)
top_authors.show(10, truncate=False)

print("\n" + "=" * 70)
print("ENCODING TYPES")
print("=" * 70)
encoding_dist.show(10, truncate=False)

# ======================================================================
# STEP 4: SAVE RESULTS (OPTIMIZED)
# ======================================================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Save metadata as parquet (most efficient)
metadata_df.write.mode("overwrite").parquet("q10_metadata.parquet")
print("Saved: q10_metadata.parquet")

# Save CSVs
books_per_year.write.mode("overwrite").option("header", "true").csv("q10_books_per_year_csv")
print("Saved: q10_books_per_year_csv/")

language_dist.write.mode("overwrite").option("header", "true").csv("q10_language_dist_csv")
print("Saved: q10_language_dist_csv/")

# Create text summary
summary_lines = []
summary_lines.append("=" * 70)
summary_lines.append("QUESTION 10: BOOK METADATA EXTRACTION - SUMMARY")
summary_lines.append("=" * 70)
summary_lines.append("")
summary_lines.append("DATASET STATISTICS:")
summary_lines.append("-" * 70)
summary_lines.append("Total books analyzed: {}".format(stats["total_books"]))
summary_lines.append("Books with title: {} ({:.1f}%)".format(
    stats["with_title"], 100 * stats["with_title"] / stats["total_books"]))
summary_lines.append("Books with author: {} ({:.1f}%)".format(
    stats["with_author"], 100 * stats["with_author"] / stats["total_books"]))
summary_lines.append("Books with year: {} ({:.1f}%)".format(
    stats["with_year"], 100 * stats["with_year"] / stats["total_books"]))
summary_lines.append("Books with language: {} ({:.1f}%)".format(
    stats["with_language"], 100 * stats["with_language"] / stats["total_books"]))
summary_lines.append("")

if stats["min_year"]:
    summary_lines.append("Year range: {} - {}".format(stats["min_year"], stats["max_year"]))
    summary_lines.append("")

summary_lines.append("TOP 5 LANGUAGES:")
summary_lines.append("-" * 70)
for row in language_dist.limit(5).collect():
    summary_lines.append("  {}: {} books".format(row["language"], row["count"]))

summary_lines.append("")
summary_lines.append("TOP 5 AUTHORS:")
summary_lines.append("-" * 70)
for row in top_authors.limit(5).collect():
    summary_lines.append("  {}: {} books".format(row["author"], row["count"]))

with open("q10_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

print("Saved: q10_summary.txt")

print("\n" + "=" * 70)
print("Question 10 Complete!")
print("=" * 70)

# Cleanup
metadata_df.unpersist()
books_per_year.unpersist()
language_dist.unpersist()
top_authors.unpersist()
encoding_dist.unpersist()

spark.stop()
