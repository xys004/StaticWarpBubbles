"""
gcp_sync.py
===========
Bridges Vertex AI Container Jobs with Google Cloud Storage.
Uploads the raw outputs (ansatz and plots) back to a bucket when the swarm succeeds.
"""

import os
from google.cloud import storage

# This will only run safely if deployed to GCP or if local gcloud is auth'd.
def is_cloud_env():
    return os.getenv("RUNNING_IN_VERTEX", "false").lower() == "true"

def upload_folder_to_gcs(bucket_name, source_folder, destination_blob_prefix):
    """Recursively uploads a directory to GCS."""
    if not is_cloud_env():
        print(f"☁️  [GCP Sync Tracker] Skpipping upload for {source_folder} because running locally.")
        return

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        for root, dirs, files in os.walk(source_folder):
            for file in files:
                local_path = os.path.join(root, file)
                # Compute relative path for GCS
                rel_path = os.path.relpath(local_path, source_folder)
                blob_path = os.path.join(destination_blob_prefix, rel_path).replace("\\", "/")
                
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(local_path)
                print(f"☁️ [GCS] Uploaded {local_path} -> gs://{bucket_name}/{blob_path}")
                
    except Exception as e:
        print(f"[!] GCS Upload Failed: {e}")

def upload_file_to_gcs(bucket_name, local_file, destination_blob_name):
    """Uploads a single file to GCS."""
    if not is_cloud_env():
        print(f"☁️  [GCP Sync Tracker] Skpipping upload for {local_file} because running locally.")
        return

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file)
        print(f"☁️ [GCS] Uploaded {local_file} -> gs://{bucket_name}/{destination_blob_name}")
    except Exception as e:
        print(f"[!] GCS Upload Failed: {e}")
