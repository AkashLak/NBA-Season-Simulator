# S3 integration is not currently configured.
# Parquet snapshots in processed/ serve as the data lake layer instead.
# To add S3 support: set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and S3_BUCKET
# in .env and implement upload_parquet_to_s3() using boto3.
