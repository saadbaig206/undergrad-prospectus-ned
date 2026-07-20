import os
import sys
from dotenv import load_dotenv
import boto3
from botocore.client import Config

load_dotenv()

def test_r2():
    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    bucket = os.getenv("R2_BUCKET_NAME", "assets")
    
    if not account_id or "your_" in account_id:
        print("ERROR: R2 credentials are not set in .env")
        return
        
    try:
        s3 = boto3.client(
            's3',
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version='s3v4')
        )
        
        # Test listing the bucket
        response = s3.list_objects_v2(Bucket=bucket, MaxKeys=5)
        print("SUCCESS: Connected to Cloudflare R2!")
        print(f"Bucket '{bucket}' exists.")
        
        if 'Contents' in response:
            print("Files in bucket:")
            for obj in response['Contents']:
                print(f" - {obj['Key']} ({obj['Size']} bytes)")
        else:
            print("Bucket is currently empty.")
            
    except Exception as e:
        print(f"ERROR connecting to R2: {e}")

if __name__ == "__main__":
    test_r2()
