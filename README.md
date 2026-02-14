# Big Data Processing: Hadoop & Spark

Distributed data processing implementations on Project Gutenberg book corpus (425+ books).

```
##  Structure
```
├── hadoop/                       # MapReduce jobs (Java) - Q1-Q9
│   ├── WordCount.java            # The latest edited WordCount code for Q5
│   ├── WordCount_q1.java         # Basic version of wordcount used in Q1,2,3
│   ├── WordCount_q4.java         # Version 2 od WordCount.Java edited for Q4. 
│
└── spark/                        # Spark analytics (Python) - Q10-Q12
    ├── load_books.py             # Data loader
    ├── q10_metadata.py           # Metadata extraction
    ├── q11_tfidf.py              # TF-IDF similarity
    └── q12_author_influence.py   # Author influence network
```

## Quick Start

**Hadoop:**
```bash
cd hadoop/
javac -classpath $(hadoop classpath) *.java
hadoop jar job.jar ClassName /input /output
```

**Spark:**
```bash
cd spark/
python3 q10_metadata.py      # Metadata extraction
python3 q11_tfidf.py          # TF-IDF similarity
python3 q12_author_influence.py  # Influence network
```

## Key Results

| Task | Dataset | Result |
|------|---------|--------|
| **Q10: Metadata** | 425 books | 96% coverage (407 books) |
| **Q11: TF-IDF** | 50 books | Similarity: 0.27-0.53 |
| **Q12: Network** | 407 authors | 14,460 influence edges |

## Tech Stack

**Hadoop:** MapReduce, Java 8+  
**Spark:** PySpark 3.5+, MLlib, Python 3.8+

## Highlights

- **Hadoop:** Word count, inverted index, TF-IDF, N-grams
- **Spark:** Metadata extraction, NLP similarity, graph analysis
- **Optimizations:** No UDFs, strategic caching, 10x speedup

Author : Nandini Sridharan (M25DE1012)

*Built with Hadoop MapReduce & Apache Spark for distributed big data processing*
