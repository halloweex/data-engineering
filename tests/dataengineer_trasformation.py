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

columns_renaming = {
    "location": "job_location",
    "size": "company_size",
    "industry": "company_industry",
    "sector": "company_sector",
    "revenue": "company_revenue",
    "company name": "company_name",
    "job description": "job_description",
    "job title": "job_title",
    "salary estimate": "salary_estimate",
    "headquarters": "headquarters",
    "easy apply": "easy_apply",
    "type of ownership": "type_of_ownership",
}

for old_col, new_col in columns_renaming.items():
    df = df.withColumnRenamed(old_col, new_col)

# Replace "-1"
df = df.withColumn("Founded", when(col("Founded") == -1, lit(None)).otherwise(col("Founded")))
df = df.withColumn("easy_apply", when(col("easy_apply") == -1, lit(None)).otherwise(col("easy_apply")))
df = df.withColumn("company_industry",
                   when(col("company_industry") == "-1", "Unknown").otherwise(col("company_industry")))
df = df.withColumn("company_size", when(col("company_size") == "-1", "Unknown").otherwise(col("company_size")))
df = df.withColumn("company_sector", when(col("company_sector") == "-1", "Unknown").otherwise(col("company_sector")))
df = df.withColumn("company_revenue", when(col("company_revenue") == "-1", "Unknown").otherwise(col("company_revenue")))
df = df.withColumn("Competitors", when(col("Competitors") == "-1", "Unknown").otherwise(col("Competitors")))
df = df.withColumn("type_of_ownership",
                   when(col("type_of_ownership") == "-1", "Unknown").otherwise(col("type_of_ownership")))

df.show()

# Dictionary mapping number words to their numeric representations
number_word_to_num = {
    'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
    'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
    'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14',
    'fifteen': '15', 'sixteen': '16', 'seventeen': '17', 'eighteen': '18',
    'nineteen': '19', 'twenty': '20'
}


# Function to replace number words with numbers
def replace_number_words(text):
    for word, num in number_word_to_num.items():
        text = text.replace(word, num)
    return text



# Create a UDF from the function
# Apply the UDF to the DataFrame
replace_number_words_udf = udf(replace_number_words, StringType())
df = df.withColumn('job_description', replace_number_words_udf(col('job_description')))

# Remove ratings from the 'company_name' column
# This regex looks for a newline followed by any number of characters (.*)
df = df.withColumn('company_name', regexp_replace('company_name', '\\n.*', ''))

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

# Count the number of nulls in 'job_experience_yrs'
null_count = \
    df.agg(count(when(col('job_experience').isNull() | (col('job_experience') == ''), True))).collect()[0][0]
print(f"Number of nulls in 'job_experience': {null_count}")

# Check if languages is mentioned in the job description and create a new column 'python'
languages = ['python', 'scala', 'java', 'sql']
for language in languages:
    df = df.withColumn(language, col('job_description').contains(language))

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


# Define UDFs for extracting min and max salaries
def extract_min_salary(value):
    try:
        return int(value.split('-')[0].replace('$', '').replace('K', '').strip()) * 1000
    except:
        return None


def extract_max_salary(value):
    try:
        return int(value.split('-')[1].split('(')[0].replace('$', '').replace('K', '').strip()) * 1000
    except:
        return None


extract_min_salary_udf = udf(extract_min_salary, IntegerType())
extract_max_salary_udf = udf(extract_max_salary, IntegerType())

# Create new columns with min, max, avg salary for each row
df = df.withColumn("min_salary", extract_min_salary_udf("salary_estimate"))
df = df.withColumn("max_salary", extract_max_salary_udf("salary_estimate"))
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

df.show()
