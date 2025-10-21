import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F

# --------------------- Init ---------------------
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# --------------------- Input ---------------------
# Load from Affiliation metadata (with row_id)
input_df = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    format="csv",
    connection_options={
        "paths": ["s3://scopus-bucket-pfe/gold/AffiliationExtractionOutput/metadata_table/"],
        "recurse": True
    },
    format_options={"withHeader": True, "separator": "|"}
).toDF()

# --------------------- Process Authors ---------------------
# Explode multiple authors in one cell
exploded_df = input_df.withColumn("author_entry", F.explode(F.split(F.col("author full names"), ";")))

# Extract author name and author ID
split_df = exploded_df.withColumn(
    "author_name",
    F.regexp_replace(F.regexp_extract(F.col("author_entry"), r"^(.*?)\s*\(\d+\)$", 1), ",", "")
).withColumn(
    "author_id",
    F.regexp_extract(F.trim(F.col("author_entry")), r"^.*?\((\d+)\)$", 1)
)

# --------------------- Output Tables ---------------------

# ✅ Author Table (one row per author)
author_table = split_df \
    .select("author_name", "author_id") \
    .dropna(subset=["author_id"]) \
    .dropDuplicates(["author_id"])

# ✅ Intermediate Mapping (row_id ↔ author_id), allowing multiple documents per author
intermediate_author_metadata = split_df \
    .select("row_id", "author_id") \
    .dropna(subset=["author_id"]) \
    .dropDuplicates(["row_id", "author_id"])

# --------------------- Write Output to S3 ---------------------
author_table.repartition(1).write.csv(
    "s3://scopus-bucket-pfe/gold/AuthorExtractionOutput/author_table/",
    mode="overwrite",
    header=True,
    sep="|"
)

intermediate_author_metadata.repartition(1).write.csv(
    "s3://scopus-bucket-pfe/gold/AuthorExtractionOutput/intermediate_author_metadata/",
    mode="overwrite",
    header=True,
    sep="|"
)

# --------------------- Done ---------------------
job.commit()





