import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col, regexp_replace, split

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read journal CSV
df = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    format="csv",
    connection_options={"paths": ["s3://scopus-bucket-pfe/bronze/Journal/"], "recurse": True},
    format_options={"withHeader": True, "separator": ",", "quoteChar": '"'}
).toDF()

# Select relevant fields
journal_df = df.select(
    col("Sourceid"), col("Title"), col("Type"),
    col("Issn").alias("Issn Online"),
    col("SJR"), col("SJR Best Quartile"), col("H index"),
    col("`Total Docs. (2024)`").alias("Total Docs 2024"),
    col("`Total Docs. (3years)`").alias("Total Docs Last 3 years"),
    col("`Total Refs.`").alias("Total Refs"),
    col("`Total Citations (3years)`").alias("Total Cites Last 3 years"),
    col("`Citable Docs. (3years)`").alias("Citable Docs Last 3 years"),
    col("`Citations / Doc. (2years)`").alias("Citations Over Docs Last 2 Years"),
    col("`Ref. / Doc.`").alias("Ref Over Docs"),
    col("%Female"), col("Overton"), col("SDG"), col("Country"),
    col("Region"), col("Publisher"), col("Categories"), col("Areas")
).dropDuplicates(["Issn Online"])

# Clean
journal_df = journal_df.select([regexp_replace(col(c), '\|', '').alias(c) for c in journal_df.columns])
journal_df = journal_df.filter(col("H index").rlike("^[0-9]+$"))

journal_df = journal_df \
    .withColumn("SJR", regexp_replace(col("SJR"), ",", ".")) \
    .withColumn("%Female", regexp_replace(col("%Female"), ",", ".")) \
    .withColumn("Ref Over Docs", regexp_replace(col("Ref Over Docs"), ",", ".")) \
    .withColumn("Citations Over Docs Last 2 Years", regexp_replace(col("Citations Over Docs Last 2 Years"), ",", ".")) \
    .withColumn("H index", col("H index").cast(IntegerType())) \
    .withColumn("Issn Online", split(col("Issn Online"), ",").getItem(0))

journal_df.repartition(1).write.csv("s3://scopus-bucket-pfe/gold/Journal Output/", mode="overwrite", header=True, sep="|", quote='')

job.commit()





