import sys
from datetime import datetime
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType


args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# Script generated for node Amazon S3 input
read_from_s3_dyF = glueContext.create_dynamic_frame.from_options(
    format_options={
        "quoteChar": '"',
        "escaper": '"',
        "withHeader": True,
        "separator": ",",
        "multiline": True,
        "optimizePerformance": False,
    },
    connection_type="s3",
    format="csv",
    connection_options={
        "paths": ["s3://vladislav-initial-s3/data/raw_data/csv/dataengineer/"],
        "recurse": True,
    },
    transformation_ctx="read_from_s3_dyF",
)

def add_source_details(rec):
    current_datetime = datetime.now()
    rec["source_date"] = current_datetime.strftime("%Y-%m-%d")
    rec["year"] = str(current_datetime.year)
    rec["month"] = str(current_datetime.month)
    rec["day"] = str(current_datetime.day)
    return rec


mapped_dyF = Map.apply(frame=read_from_s3_dyF, f=add_source_details)


# Script generated for node Change Schema
change_schema_dyF = ApplyMapping.apply(
    frame=mapped_dyF,
    mappings=[
        ("job title", "string", "job_title", "string"),
        ("salary estimate", "string", "salary_estimate", "string"),
        ("job description", "string", "job_description", "string"),
        ("rating", "string", "rating", "string"),
        ("company name", "string", "company_name", "string"),
        ("location", "string", "location", "string"),
        ("headquarters", "string", "headquarters", "string"),
        ("size", "string", "size", "string"),
        ("founded", "string", "founded", "string"),
        ("type of ownership", "string", "type_of_ownership", "string"),
        ("industry", "string", "industry", "string"),
        ("sector", "string", "sector", "string"),
        ("revenue", "string", "revenue", "string"),
        ("competitors", "string", "competitors", "string"),
        ("easy apply", "string", "easy_apply", "string"),
        ("source_date", "string", "source_date", "string"),
        ("year", "string", "year", "int"),
        ("month", "string", "month", "int"),
        ("day", "string", "day", "int"),
    ],
    transformation_ctx="change_schema_dyF",
)

# Script generated for node Amazon S3
save_dyF_to_s3 = glueContext.getSink(
    path="s3://vladislav-initial-s3/data/raw_data_parquet/dataengineer/",
    connection_type="s3",
    updateBehavior="UPDATE_IN_DATABASE",
    partitionKeys=["year", "month", "day"],
    enableUpdateCatalog=True,
    transformation_ctx="save_dyF_to_s3",
)
save_dyF_to_s3.setCatalogInfo(
    catalogDatabase="de_project_db", catalogTableName="dataengineer_parquet"
)
save_dyF_to_s3.setFormat("glueparquet", compression="snappy")
save_dyF_to_s3.writeFrame(change_schema_dyF)
job.commit()
