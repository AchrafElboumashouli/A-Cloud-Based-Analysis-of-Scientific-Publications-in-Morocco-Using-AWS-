import sys, re, unicodedata
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from pyspark.sql.window import Window

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# ------------------- Load Mapping JSONs -------------------
def read_json(path, schema):
    return spark.read.schema(schema).option("multiline", True).json(path)

school_schema = StructType([
    StructField("school", StringType(), True),
    StructField("university", StringType(), True),
    StructField("city", StringType(), True),
    StructField("variations", ArrayType(StringType()), True)
])
uni_schema = StructType([
    StructField("standardized_uni", StringType(), True),
    StructField("variations", ArrayType(StringType()), True)
])
cities_schema = StructType([
    StructField("standardized_city", StringType(), True),
    StructField("variations", ArrayType(StringType()), True)
])

school_mapping_list = read_json("s3://scopus-bucket-pfe/Resources/school.json", school_schema).rdd.map(lambda row: {
    "school": row.school,
    "university": row.university,
    "city": row.city,
    "variations": [v.lower() for v in row.variations or []]
}).collect()
uni_mapping = read_json("s3://scopus-bucket-pfe/Resources/university.json", uni_schema).rdd.map(lambda row: (row.standardized_uni, row.variations or [])).collectAsMap()
cities_mapping = read_json("s3://scopus-bucket-pfe/Resources/city.json", cities_schema).rdd.map(lambda row: (row.standardized_city, row.variations or [])).collectAsMap()

# ------------------- Input -------------------
input_df = glueContext.create_dynamic_frame.from_options(
    format_options={"withHeader": True, "separator": "|"},
    connection_type="s3",
    format="csv",
    connection_options={"paths": ["s3://scopus-bucket-pfe/silver/"], "recurse": True}
).toDF()

print("Columns loaded from input CSV:", input_df.columns)

input_df = input_df.withColumn("row_id", monotonically_increasing_id())
input_df = input_df.withColumn("Affiliations", when(col("Affiliations").isNull(), "").otherwise(col("Affiliations")))
exploded_df = input_df.withColumn("Affiliation", explode(split(col("Affiliations"), ";")))

# ------------------- Normalize Functions -------------------
def normalize(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"[.,;:/()\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")

@udf(StringType())
def normalize_udf(text): return normalize(text)

def extract_city(affiliation):
    normalized = normalize(affiliation)
    for std, variations in cities_mapping.items():
        if any(v.lower() in normalized for v in variations):
            return std
    return None

def extract_country(affiliation):
    affiliation_lower = affiliation.lower()
    if "morocco" in affiliation_lower or "maroc" in affiliation_lower:
        return "Morocco"
    parts = [p.strip() for p in affiliation.split(',')]
    return parts[-1] if parts else ""

def match_school_university(affiliation):
    norm = normalize(affiliation)
    city = extract_city(affiliation)
    for s in school_mapping_list:
        if city and s['city'].lower() != city.lower(): continue
        if any(normalize(v) in norm for v in s['variations']):
            return s['school'], s['university']
    return None, None

def extract_school(aff): s, _ = match_school_university(aff); return s
def extract_university(aff):
    _, u = match_school_university(aff)
    if u: return u
    norm = normalize(aff)
    for std, vars in uni_mapping.items():
        if any(normalize(v) in norm for v in vars): return std
    return None

def extract_laboratoire(aff):
    for part in aff.split(','):
        norm = normalize(part)
        if any(k in norm for k in ["laboratoire", "research", "centre", "institut", "team", "lab"]):
            return part.strip()
    return None

schema = StructType([
    StructField("school", StringType(), True),
    StructField("university", StringType(), True),
    StructField("city", StringType(), True),
    StructField("country", StringType(), True),
    StructField("laboratoire", StringType(), True)
])

@udf(schema)
def process(aff):
    try:
        return (
            extract_school(aff),
            extract_university(aff),
            extract_city(aff),
            extract_country(aff),
            extract_laboratoire(aff)
        )
    except:
        return (None, None, None, None, None)

# ------------------- Processing -------------------
exploded_df = exploded_df.withColumn("aff_clean", normalize_udf(col("Affiliation")))
processed_df = exploded_df.withColumn("processed", process(col("Affiliation"))).select(
    "*",
    col("processed.school"), col("processed.university"),
    col("processed.city"), col("processed.country"),
    col("processed.laboratoire")
).drop("processed")

# Only Morocco
target_df = processed_df.filter(col("country") == "Morocco")

# Generate join_key
target_df = target_df.withColumn("join_key", sha2(col("aff_clean"), 256))
exploded_df = exploded_df.withColumn("join_key", sha2(col("aff_clean"), 256))

# Build affiliation table (include aff_clean for human readability)
aff_table = target_df.select(
    "join_key",
    "aff_clean",  
    "university", "laboratoire", "school", "city", "country"
).dropDuplicates(["university", "laboratoire", "school"]) \
 .withColumn("affiliate_id", row_number().over(Window.orderBy("university", "laboratoire", "school")))

# Enrich original
with_fk = exploded_df.join(aff_table, on="join_key", how="left")

# Filter only valid input columns (ignore auto-generated like col86–col101)
valid_cols = [c for c in input_df.columns if c not in {"Affiliations", "row_id"} and not c.startswith("col")]

# Group back by row_id for metadata
grouped = with_fk.groupBy("row_id").agg(
    collect_list("affiliate_id").alias("affiliate_ids"),
    *[first(col(c)).alias(c) for c in valid_cols]
)

# ------------------- Final Output -------------------
def clean(df):
    for c in df.columns:
        df = df.withColumn(c, regexp_replace(col(c).cast(StringType()), '"', ''))
    return df

aff_table = clean(aff_table)
final_meta = clean(grouped)
intermediate_affiliation_metadata = clean(with_fk.select("row_id", "affiliate_id").distinct())

aff_table.repartition(1).write.csv("s3://scopus-bucket-pfe/gold/AffiliationExtractionOutput/affiliation_table/", mode="overwrite", header=True, sep="|")
final_meta.repartition(1).write.csv("s3://scopus-bucket-pfe/gold/AffiliationExtractionOutput/metadata_table/", mode="overwrite", header=True, sep="|")
intermediate_affiliation_metadata.repartition(1).write.csv("s3://scopus-bucket-pfe/gold/AffiliationExtractionOutput/intermediate_affiliation_metadata/", mode="overwrite", header=True, sep="|")

job.commit()


