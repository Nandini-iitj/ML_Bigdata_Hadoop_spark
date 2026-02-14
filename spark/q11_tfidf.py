from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import HashingTF, IDF, StopWordsRemover, Tokenizer
import numpy as np

# Optimized Spark configuration for 50 books
spark = SparkSession.builder \
    .appName("Q11 TF-IDF") \
    .master("local[4]") \
    .config("spark.driver.memory", "3g") \
    .config("spark.executor.memory", "3g") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.default.parallelism", "4") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("=" * 70)
print("Question 11: TF-IDF and Book Similarity")
print("=" * 70)

# ======================================================================
# STEP 1: LOAD BOOKS (Optimized read)
# ======================================================================

print("\nLoading books...")
books_df = spark.read.parquet("books_df.parquet") \
    .select("file_name", "text") \
    .limit(50) \
    .repartition(4)  # Evenly distribute across 4 partitions

print("Loaded 50 books")

# ======================================================================
# STEP 2: PREPROCESSING (Streamlined - No Header Removal)
# ======================================================================

print("\n" + "=" * 70)
print("PREPROCESSING")
print("=" * 70)

# Single-pass cleaning: lowercase + remove punctuation
print("Cleaning text...")
cleaned_df = books_df.withColumn(
    "clean_text",
    F.lower(F.regexp_replace(F.col("text"), "[^a-zA-Z\\s]+", " "))
)

# Tokenize
print("Tokenizing...")
tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
tokenized_df = tokenizer.transform(cleaned_df)

# Remove stop words
print("Removing stop words...")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
filtered_df = remover.transform(tokenized_df) \
    .select("file_name", "filtered_words")  # Drop unnecessary columns

print("Preprocessing complete!")

# ======================================================================
# STEP 3: TF-IDF CALCULATION (Optimized)
# ======================================================================

print("\n" + "=" * 70)
print("TF-IDF CALCULATION")
print("=" * 70)

print("Computing Term Frequency...")
hashingTF = HashingTF(
    inputCol="filtered_words", 
    outputCol="tf_features", 
    numFeatures=2000  # Balanced: not too small, not too large
)
tf_df = hashingTF.transform(filtered_df)

print("Computing Inverse Document Frequency...")
idf = IDF(inputCol="tf_features", outputCol="tfidf_features")
idf_model = idf.fit(tf_df)
tfidf_df = idf_model.transform(tf_df) \
    .select("file_name", "tfidf_features")  # Only keep what we need

print("TF-IDF complete!")

# ======================================================================
# STEP 4: SIMILARITY CALCULATION (Optimized - Collect Once)
# ======================================================================

print("\n" + "=" * 70)
print("SIMILARITY CALCULATION")
print("=" * 70)

# Collect all vectors at once (efficient with 50 books)
print("Collecting book vectors...")
book_vectors = tfidf_df.collect()

# Extract target book (first one)
target_book = book_vectors[0]["file_name"]
target_vector = book_vectors[0]["tfidf_features"].toArray()

print("Target book: {}".format(target_book))
print("Computing cosine similarities...")

# Vectorized similarity computation
def compute_similarity(vec1, vec2_sparse):
    vec2 = vec2_sparse.toArray()
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return float(dot_product / norm_product) if norm_product > 0 else 0.0

# Compute all similarities in one pass
similarities = [
    (book["file_name"], compute_similarity(target_vector, book["tfidf_features"]))
    for book in book_vectors
    if book["file_name"] != target_book
]

# Sort and get top 5
similarities.sort(key=lambda x: x[1], reverse=True)
top_5 = similarities[:5]

# ======================================================================
# STEP 5: DISPLAY RESULTS
# ======================================================================

print("\n" + "=" * 70)
print("TOP 5 SIMILAR BOOKS TO: {}".format(target_book))
print("=" * 70)

for rank, (book_name, similarity_score) in enumerate(top_5, 1):
    print("{}. {} - Similarity: {:.4f}".format(rank, book_name, similarity_score))

# ======================================================================
# STEP 6: SAVE RESULTS
# ======================================================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Save TF-IDF features
tfidf_df.write.mode("overwrite").parquet("q11_tfidf_features.parquet")
print("Saved: q11_tfidf_features.parquet")

# Save similarity results as text
with open("q11_similarity_results.txt", "w") as f:
    f.write("=" * 70 + "\n")
    f.write("TF-IDF BOOK SIMILARITY ANALYSIS\n")
    f.write("=" * 70 + "\n\n")
    f.write("Dataset: 50 books\n")
    f.write("Target book: {}\n".format(target_book))
    f.write("TF-IDF features: 2000\n\n")
    f.write("Top 5 Most Similar Books:\n")
    f.write("-" * 70 + "\n")
    for rank, (book_name, similarity_score) in enumerate(top_5, 1):
        f.write("{}. {} - Similarity: {:.4f}\n".format(rank, book_name, similarity_score))

print("Saved: q11_similarity_results.txt")

print("\n" + "=" * 70)
print("Question 11 Complete!")
print("=" * 70)

spark.stop()
