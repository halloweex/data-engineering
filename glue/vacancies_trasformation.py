import sys
import re
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame

from pyspark.sql.functions import when, col, regexp_extract, coalesce, split, udf, regexp_replace, lit, lower, count
from pyspark.sql.types import IntegerType, StringType

args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# Script generated for node Amazon S3
S3_input_parquet = glueContext.create_dynamic_frame.from_options(
    format_options={},
    connection_type="s3",
    format="parquet",
    connection_options={
        "paths": ["s3://vladislav-initial-s3/data/raw_data_parquet/dataengineer/"],
        "recurse": True,
    },
    transformation_ctx="S3_input_parquet",
)

df = S3_input_parquet.toDF()


number_word_to_num = {
    'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
    'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
    'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14',
    'fifteen': '15', 'sixteen': '16', 'seventeen': '17', 'eighteen': '18',
    'nineteen': '19', 'twenty': '20'
}

# Function to replace number words with numbers
def replace_number_words(text):
    def replace_match(match):
        word = match.group(0)
        return number_word_to_num.get(word, word)  # Return the number or the original word if not found

    # Use a word boundary \b to ensure only whole words are matched
    pattern = r'\b(' + '|'.join(re.escape(key) for key in number_word_to_num.keys()) + r')\b'
    return re.sub(pattern, replace_match, text)

# Create a UDF from the function
# Apply the UDF to the DataFrame
replace_number_words_udf = udf(replace_number_words, StringType())
df = df.withColumn('job_description', replace_number_words_udf(col('job_description')))


# Remove ratings from the 'company name' column
# This regex looks for a newline followed by any number of characters (.*)
df = df.withColumn('company_name', regexp_replace('company_name', '\\n.*', ''))


# Define the regex pattern to extract job experience
experience_pattern = (
    r'\b(\d+)'  # Capture an initial numeric value
    r'(\s*-\s*\d+)?'  # Optionally capture a range (' - ' followed by another number)
    r'\s*(to|\+)?\s*'  # Optionally capture 'to' or '+' indicating a range or more
    r'(\d*)'  # Optionally capture the second part of the range
    r'\s*(?:years?|yrs?|yr)\b'  # Match 'years', 'yrs', or 'yr'
    r'(\s+of\s+experience)?\b'  # Optionally match ' of experience'
)

# Apply this pattern in your existing PySpark script:
df = df.withColumn('job_experience', regexp_extract(col('job_description'), experience_pattern, 1))

def extract_min_exp(exp_range):
    if exp_range:
        numbers = re.findall(r'\d+', exp_range)
        int_numbers = [int(num) for num in numbers if int(num) <= 20]
        return min(int_numbers) if int_numbers else None
    return None


extract_min_experience_udf = udf(extract_min_exp, IntegerType())

# Apply the UDF to the DataFrame to create a new column 'min_years_experience'
df = df.withColumn('min_years_experience',
                   extract_min_experience_udf(regexp_extract(col('job_description'), experience_pattern, 0)))



# Check if languages is mentioned in the job description and create a new column 'python'
languages = ['python', 'scala', 'java', 'sql']
for language in languages:
    df = df.withColumn(language, lower(col('job_description')).contains(language))

# Define a regular expression pattern for AWS and Amazon Web Services with word boundaries
aws_pattern = r'\b(aws|amazon web services)\b'

# Update the column 'aws' with a more accurate condition using the regex pattern
# Ensure the job description is lowercased before applying the regex
df = df.withColumn('aws', when(
    lower(col('job_description')).rlike(aws_pattern), True).otherwise(False))

df = df.withColumn('azure', when(
    lower(col('job_description')).contains('azure') | lower(col('job_description')).contains('microsoft azure'),
    True).otherwise(False))
df = df.withColumn('gcp', when(
    lower(col('job_description')).contains('gcp') | lower(col('job_description')).contains('google cloud platform'),
    True).otherwise(False))

def extract_salary(value, position):
    try:
        # Remove '$' and 'K', then split by '-' and get the part based on position
        salary_part = value.replace('$', '').replace('K', '').split('-')[position].split('(')[0].strip()
        return int(salary_part) * 1000
    except:
        return None

# Define UDFs for extracting min and max salaries
extract_salary_udf = udf(lambda value, pos: extract_salary(value, pos), IntegerType())

# Create new columns with min, max, avg salary for each row
df = df.withColumn("min_salary", extract_salary_udf("salary_estimate", lit(0)))
df = df.withColumn("max_salary", extract_salary_udf("salary_estimate", lit(1)))
df = df.withColumn("avg_salary", (col("min_salary") + col("max_salary")) / 2)

# Regular expression for job levels
job_level_pattern = (
    r'\b(entry-level|junior|jr\.?|mid-level|senior|sr\.?|executive|exec\.?|lead|regular)\b'
)


def extract_job_level(description):
    match = re.search(job_level_pattern, description, re.IGNORECASE)
    if match:
        level = match.group(1).lower()
        mapping = {
            'entry-level': 'Entry-Level',
            'junior': 'Junior', 'jr': 'Junior', 'jr.': 'Junior',
            'mid-level': 'Mid-Level',
            'senior': 'Senior', 'sr': 'Senior', 'sr.': 'Senior',
            'executive': 'Executive', 'exec': 'Executive', 'exec.': 'Executive',
            'lead': 'Lead',
            'regular': 'Regular'  # Added mapping for Regular
        }
        return mapping.get(level, None)
    return None


# Apply the UDF to create a new column 'job_level'
extract_job_level_udf = udf(extract_job_level, StringType())
df = df.withColumn('job_level', extract_job_level_udf(col('job_description')))

df = df.withColumn(
    'job_level',
    when(
        col('job_level').isNull(),
        when(col('min_years_experience') <= 3, 'Regular').otherwise('Senior')
    ).otherwise(col('job_level'))
)

# Detect mentions of 'remote' and 'relocation' in job descriptions and create corresponding boolean columns
df = df.withColumn(
    'is_remote',
    when(lower(col('job_description')).contains('remote'), True)
    .otherwise(False)
)

df = df.withColumn(
    'is_relocation',
    when(lower(col('job_description')).contains('relocation'), True)
    .otherwise(False)
)

df_final = df.select('job_title',
'salary_estimate',
'rating',
'company_name',
'location',
'headquarters',
'size',
'founded',
'type_of_ownership',
'industry',
'sector',
'revenue',
'competitors',
'easy_apply',
'source_date',
'job_experience',
'min_years_experience',
'python',
'scala',
'java',
'sql',
'aws',
'azure',
'gcp',
'min_salary',
'max_salary',
'avg_salary',
    'job_level',
'is_remote',
'is_relocation')

S3_input_parquet = DynamicFrame.fromDF(df_final, glueContext, "S3_input_parquet")

# Script generated for node Amazon S3
S3_output_parquet = glueContext.getSink(
    path="s3://vladislav-initial-s3/data/output_data_parquet/dataengineer/",
    connection_type="s3",
    updateBehavior="UPDATE_IN_DATABASE",
    partitionKeys=[],
    enableUpdateCatalog=True,
    transformation_ctx="AmazonS3_output_data_parquet",
)
S3_output_parquet.setCatalogInfo(
    catalogDatabase="de_project_db", catalogTableName="output_table_dataengineer"
)
S3_output_parquet.setFormat("glueparquet", compression="snappy")
S3_output_parquet.writeFrame(S3_input_parquet)
job.commit()
