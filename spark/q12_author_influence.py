from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

spark = SparkSession.builder \
    .appName("Q12 Author Influence") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 70)
print("Question 12: Author Influence Network")
print("=" * 70)

# ======================================================================
# OPTIMIZED: Load and extract metadata in ONE STEP
# ======================================================================

print("\nLoading books and extracting metadata...")

# Single pass: load + extract + filter
metadata_df = spark.read.parquet("books_df.parquet") \
    .select("file_name", "text") \
    .withColumn("author", 
        F.regexp_extract(F.col("text"), r'Author:\s*(.+?)(?:\r?\n)', 1)) \
    .withColumn("year_str", 
        F.regexp_extract(F.col("text"), r'Release Date:.*?(\d{4})', 1)) \
    .withColumn("year", 
        F.when(F.col("year_str") != "", F.col("year_str").cast(IntegerType()))
         .otherwise(None)) \
    .filter((F.col("author") != "") & F.col("year").isNotNull()) \
    .select("file_name", "author", "year") \
    .distinct() \
    .cache()

# Single count operation
books_with_metadata = metadata_df.count()
print("Books with valid metadata: {}".format(books_with_metadata))

print("\nSample metadata:")
metadata_df.show(10, truncate=False)

# ======================================================================
# BUILD INFLUENCE NETWORK
# ======================================================================

print("\n" + "=" * 70)
print("BUILDING INFLUENCE NETWORK")
print("=" * 70)

TIME_WINDOW = 10
print("Time window: {} years".format(TIME_WINDOW))

# Self-join
books_a = metadata_df.alias("a")
books_b = metadata_df.alias("b")

influence_edges = books_a.join(books_b,
    (F.col("a.author") != F.col("b.author")) &
    (F.col("a.year") < F.col("b.year")) &
    (F.col("b.year") - F.col("a.year") <= TIME_WINDOW)
) \
    .select(
        F.col("a.author").alias("influencer"),
        F.col("b.author").alias("influenced")
    ) \
    .distinct() \
    .cache()

total_edges = influence_edges.count()
print("Total influence edges: {}".format(total_edges))

# ======================================================================
# COMPUTE DEGREES
# ======================================================================

print("\n" + "=" * 70)
print("COMPUTING METRICS")
print("=" * 70)

out_degree = influence_edges.groupBy("influencer").count() \
    .withColumnRenamed("count", "out_degree") \
    .withColumnRenamed("influencer", "author")

in_degree = influence_edges.groupBy("influenced").count() \
    .withColumnRenamed("count", "in_degree") \
    .withColumnRenamed("influenced", "author")

all_authors = metadata_df.select("author").distinct()

author_metrics = all_authors \
    .join(in_degree, "author", "left") \
    .join(out_degree, "author", "left") \
    .fillna(0) \
    .cache()

# ======================================================================
# DISPLAY
# ======================================================================

print("\n" + "=" * 70)
print("TOP 5 BY OUT-DEGREE (Most Influential)")
print("=" * 70)
author_metrics.orderBy(F.desc("out_degree")).show(5, truncate=False)

print("\n" + "=" * 70)
print("TOP 5 BY IN-DEGREE (Most Influenced)")
print("=" * 70)
author_metrics.orderBy(F.desc("in_degree")).show(5, truncate=False)

# ======================================================================
# SAVE
# ======================================================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

influence_edges.write.mode("overwrite").parquet("q12_influence_edges.parquet")
author_metrics.write.mode("overwrite").parquet("q12_author_metrics.parquet")

with open("q12_summary.txt", "w") as f:
    f.write("=" * 70 + "\n")
    f.write("AUTHOR INFLUENCE NETWORK SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    f.write("Books with metadata: {}\n".format(books_with_metadata))
    f.write("Time window: {} years\n".format(TIME_WINDOW))
    f.write("Total influence edges: {}\n\n".format(total_edges))
    f.write("TOP 5 INFLUENTIAL (Out-Degree):\n")
    for row in author_metrics.orderBy(F.desc("out_degree")).limit(5).collect():
        f.write("  {}: {}\n".format(row["author"], row["out_degree"]))
    f.write("\nTOP 5 INFLUENCED (In-Degree):\n")
    for row in author_metrics.orderBy(F.desc("in_degree")).limit(5).collect():
        f.write("  {}: {}\n".format(row["author"], row["in_degree"]))

print("Saved: q12_summary.txt")

print("\n" + "=" * 70)
print("Question 12 Complete!")
print("=" * 70)

metadata_df.unpersist()
influence_edges.unpersist()
author_metrics.unpersist()

spark.stop()
