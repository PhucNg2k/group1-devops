from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import os
import logging
import pandas as pd
from functools import reduce
import boto3
from botocore.exceptions import ClientError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# S3 Configuration
SILVER_BUCKET = "silver-layer-devops-g125"

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define paths
raw_path = os.path.join(SCRIPT_DIR, 'raw', '*.json')
mlflow_schema_path = '../MLFlow/data/Botnet-Friday-no-metadata.parquet'  # MLFlow schema
silver_path = f's3a://{SILVER_BUCKET}/silver'  # S3 silver directory


# Cache for MLFlow schema
_mlflow_schema_cache = None

def get_mlflow_schema(spark):
    """Get MLFlow schema, inferring it only once and caching the result"""
    global _mlflow_schema_cache
    if _mlflow_schema_cache is None:
        try:
            logger.info(f"Reading MLFlow schema from: {mlflow_schema_path}")
            df = spark.read.parquet(mlflow_schema_path)
            _mlflow_schema_cache = df.schema
            logger.info(f"Inferred MLFlow schema with {len(_mlflow_schema_cache.fields)} fields")
        except Exception as e:
            logger.error(f"Error inferring MLFlow schema: {str(e)}")
            raise
    return _mlflow_schema_cache

def validate_mlflow_schema(df, mlflow_schema):
    """Validate DataFrame schema against MLFlow schema"""
    actual_fields = set(field.name for field in df.schema.fields)
    mlflow_fields = set(field.name for field in mlflow_schema.fields)
    
    extra_fields = actual_fields - mlflow_fields
    if extra_fields:
        logger.warning(f"Found extra fields in data: {extra_fields}")
    
    missing_fields = mlflow_fields - actual_fields
    if missing_fields:
        logger.warning(f"Missing required MLFlow fields: {missing_fields}")
        return False
    
    # Check field types
    for mlflow_field in mlflow_schema.fields:
        actual_field = next((f for f in df.schema.fields if f.name == mlflow_field.name), None)
        if actual_field.dataType != mlflow_field.dataType:
            logger.warning(f"Type mismatch for field {mlflow_field.name}: expected {mlflow_field.dataType}, got {actual_field.dataType}")
            return False
    
    return True

def validate_mlflow_records(df, mlflow_fields):
    """Validate records for non-null MLFlow fields and return valid DataFrame"""
    try:
        # Check for nulls in required fields
        conditions = [col(field).isNotNull() for field in mlflow_fields]
        if not conditions:
            return df, 0
        
        valid_df = df.filter(reduce(lambda x, y: x & y, conditions))
        
        # Log sample invalid records (up to 5)
        invalid_df = df.filter(~reduce(lambda x, y: x & y, conditions)).limit(5)
        if invalid_df.count() > 0:
            #logger.warning("Sample invalid records:")
            for row in invalid_df.collect():
                #logger.warning(f"  {row.asDict()}")
                pass
        
        return valid_df, valid_df.count()
    except Exception as e:
        logger.error(f"Error in record validation: {str(e)}")
        return df, 0

def create_spark_session():
    """Create and configure Spark session"""
    try:
        return SparkSession.builder \
            .appName("MLFlowDataPipeline") \
            .config("spark.jars.packages", 
                    "org.apache.hadoop:hadoop-aws:3.3.4,"
                    "org.apache.hadoop:hadoop-client:3.3.4,"
                    "software.amazon.awssdk:bundle:2.20.18,"
                    "commons-cli:commons-cli:1.5.0") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", 
                    "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
            .config("spark.sql.shuffle.partitions", "12") \
            .config("spark.executor.cores", "4") \
            .config("spark.executor.memory", "8g") \
            .config("spark.driver.memory", "8g") \
            .getOrCreate()
    except Exception as e:
        logger.error(f"Failed to create Spark session: {str(e)}")
        raise

def check_prerequisites():
    """Check file existence and S3 connectivity"""
    try:
        # Check raw JSON files
        if not os.path.exists(os.path.dirname(raw_path)):
            logger.error(f"Raw data directory does not exist: {os.path.dirname(raw_path)}")
            return False
        
        # Check S3 connectivity
        s3 = boto3.client('s3')
        s3.head_bucket(Bucket=SILVER_BUCKET)
        logger.info(f"S3 bucket {SILVER_BUCKET} is accessible")
        return True
    except ClientError as e:
        logger.error(f"S3 bucket {SILVER_BUCKET} is not accessible: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Prerequisite check failed: {str(e)}")
        return False

def process_data(spark):
    try:
        logger.info("Starting data processing...")
        
        # Get MLFlow schema first
        logger.info("Reading MLFlow schema...")
        mlflow_df = spark.read.parquet(mlflow_schema_path)
        mlflow_schema = mlflow_df.schema
        mlflow_columns = [field.name for field in mlflow_schema.fields]
        logger.info(f"MLFlow schema has {len(mlflow_columns)} columns")
        
        # Read JSON with MLFlow schema
        logger.info(f"Reading JSON files from: {raw_path}")
        df = spark.read.schema(mlflow_schema).json(raw_path).cache()
        
        # Select only columns that match MLFlow schema
        #df = df.select(*[col(c) for c in mlflow_columns if c in df.columns])
        
        # Log what we got
        total_count = df.count()
        logger.info(f"Successfully read {total_count:,} records")
        logger.info(f"Columns in processed data: {df.columns}")
        
        # Write to parquet
        parquet_path = f"{silver_path}/mlflow_aligned_data"
        df.write \
            .format("parquet") \
            .mode("overwrite") \
            .save(parquet_path)
        
        logger.info(f"Wrote {total_count:,} records to {parquet_path}")
        df.unpersist()
        
        logger.info("Data processing complete!")
        
    except Exception as e:
        logger.error(f"Error in process_data: {str(e)}")
        raise

def main():
    """Main entry point for the pipeline"""
    try:
        logger.info("Starting MLFlow data pipeline...")
        
        # Initialize Spark
        spark = create_spark_session()
        
        # Process data
        process_data(spark)
        
    except Exception as e:
        logger.error(f"Pipeline error occurred: {type(e).__name__}: {str(e)}")
        sys.exit(1)
    finally:
        if 'spark' in locals():
            spark.stop()
            logger.info("Pipeline shutdown complete")

if __name__ == "__main__":
    main()