import sys, re
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, regexp_replace

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read metadata
df = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    format="csv",
    connection_options={"paths": ["s3://scopus-bucket-pfe/gold/AffiliationExtractionOutput/metadata_table/"], "recurse": True},
    format_options={"withHeader": True, "separator": "|"}
).toDF()

# Remove prefix and extract doc_id
doc_df = df.select(
    regexp_replace(col("eid"), r"^2-s2\.0-", "").alias("doc_id"),
    "doi", "title", "document type", "volume", "issue", "art_no",
    "page start", "page end", "page count", "link", "abstract", "issn",
    "language of original document", "author keywords", "references"
).dropDuplicates(["doc_id"]).filter(col("doc_id").cast("double").isNotNull())

# Minimal metadata without document details (retain row_id)
metadata_df = df.select(
    regexp_replace(col("eid"), r"^2-s2\.0-", "").alias("doc_id"),
    *[col(c) for c in df.columns if c not in [
        "title", "volume", "issue", "art_no", "page start", "page end",
        "page count", "link", "abstract", "issn", "language of original document",
        "doi", "isbn", "references", "eid", "document type", "author keywords"
    ]]
).filter(col("doc_id").cast("double").isNotNull())

# Clean
for c in doc_df.columns: doc_df = doc_df.withColumn(c, regexp_replace(col(c), '"', ''))
for c in metadata_df.columns: metadata_df = metadata_df.withColumn(c, regexp_replace(col(c), '"', ''))

# Save
doc_df.repartition(1).write.csv("s3://scopus-bucket-pfe/gold/DocumentExtractionOutput/document_table/", mode="overwrite", header=True, sep="|")
metadata_df.repartition(1).write.csv("s3://scopus-bucket-pfe/gold/MetadataExtractionOutput/metadata_table/", mode="overwrite", header=True, sep="|")

job.commit()






