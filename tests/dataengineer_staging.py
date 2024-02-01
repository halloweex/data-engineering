from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, regexp_extract, coalesce, split, udf, regexp_replace, lit, lower, count
from pyspark.sql.types import IntegerType, StringType
import re

# Start Spark session
spark = SparkSession.builder.getOrCreate()

# Read the CSV file
df = spark.read.option("delimiter", ",") \
    .option("quote", "\"") \
    .option("multiLine", True) \
    .option("escape", "\"") \
    .csv("/Users/vladislav/Documents/glue-project/raw_data/csv/dataengineer/DataEngineer.csv",
        header=True, inferSchema=True)

# Show initial data
df.show()
