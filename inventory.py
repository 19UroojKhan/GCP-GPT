
import streamlit as st
import os
import json
import tempfile
from datetime import datetime
from google.cloud import asset_v1
from google.protobuf.json_format import MessageToDict
import boto3
import pinecone
from ingestion_script import LangchainPineconeLoader

# ------------------------ #
# Setup
# ------------------------ #
st.set_page_config(page_title="GCP Copilot Automation", layout="wide")
st.title("🛠️ GCP Copilot Automation + Pinecone Ingestion")

# S3 & Pinecone Config
bucket_name = "datacrux-dev"
log_key = "copilot/ingestion_log.json"
prefix = "copilot/"
s3 = boto3.client("s3")

# ------------------------ #
# Helper: Load/Save Log from S3
# ------------------------ #
def load_ingestion_log():
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            s3.download_file(bucket_name, log_key, tmp.name)
            with open(tmp.name, "r") as f:
                return json.load(f)
    except s3.exceptions.ClientError:
        return {}

def save_ingestion_log(log_data):
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as tmp:
        json.dump(log_data, tmp, indent=2)
        tmp.flush()
        s3.upload_file(tmp.name, bucket_name, log_key)

# Load current ingestion log from S3
ingestion_log = load_ingestion_log()

# ------------------------ #
# Step 1: GCP Inventory Upload to S3 + Local Download
# ------------------------ #
uploaded_key = st.file_uploader("📤 Upload GCP Service Account JSON", type="json")
project_id = st.text_input("🔑 Enter GCP Project ID")

if uploaded_key and project_id:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(uploaded_key.read())
        service_key_path = tmp.name

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_key_path
    st.success("✅ Service account saved and configured")

    if st.button("🚀 Fetch GCP Asset Inventory"):
        st.write("🔄 Fetching assets from GCP...")

        def list_assets(project_id):
            client = asset_v1.AssetServiceClient()
            parent = f"projects/{project_id}"
            request = {"parent": parent, "content_type": asset_v1.ContentType.RESOURCE}
            assets = {"assets": []}
            for asset in client.list_assets(request=request):
                assets["assets"].append(MessageToDict(asset._pb))
            return assets

        try:
            assets_json = list_assets(project_id)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"gcp_inventory_{timestamp}.json"
            local_path = os.path.join(tempfile.gettempdir(), filename)

            # Save locally
            with open(local_path, 'w') as f:
                json.dump(assets_json, f, indent=4)

            # Download to user's machine
            with open(local_path, 'rb') as f:
                st.download_button(
                    label="📥 Download GCP Inventory JSON",
                    data=f,
                    file_name=filename,
                    mime="application/json"
                )

            # Upload temporarily to S3
            s3_key = f"{prefix}{filename}"
            s3.upload_file(local_path, bucket_name, s3_key)
            st.success(f"✅ Temporary upload to `s3://{bucket_name}/{s3_key}` done.")

        except Exception as e:
            st.error(f"❌ Failed to fetch or upload asset inventory: {e}")

# ------------------------ #
# Step 2: Pinecone Ingestion
# ------------------------ #
st.markdown("---")
st.subheader("🔁 Ingest JSON Inventory into Pinecone")

try:
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    json_files = sorted(
        [obj for obj in objects.get("Contents", []) if obj["Key"].endswith(".json") and "ingestion_log" not in obj["Key"]],
        key=lambda x: x["LastModified"],
        reverse=True
    )
except Exception as e:
    st.error(f"❌ Failed to list S3 objects: {e}")
    json_files = []

if json_files:
    latest_file_key = json_files[0]["Key"]
    latest_file_name = os.path.basename(latest_file_key)

    st.write(f"📄 Latest JSON in S3: `{latest_file_name}`")

    if latest_file_name in ingestion_log:
        st.info(f"✅ This file has **already been ingested** into Pinecone index: `{ingestion_log[latest_file_name]}`. Skipping ingestion.")
    else:
        index_name = st.text_input("📇 Enter a new Pinecone Index Name for ingestion")

        if index_name and st.button("📥 Start Ingestion"):
            try:
                # Download file from S3
                latest_local = os.path.join(tempfile.gettempdir(), latest_file_name)
                s3.download_file(bucket_name, latest_file_key, latest_local)

                # Ingest into Pinecone
                st.info(f"🚀 Ingesting `{latest_file_name}` into Pinecone index `{index_name}`...")
                loader = LangchainPineconeLoader(
                    bucket_name=bucket_name,
                    directory_path=prefix,
                    index_name=index_name
                )
                loader.load_and_index()

                # Update and upload ingestion log to S3
                ingestion_log[latest_file_name] = index_name
                save_ingestion_log(ingestion_log)

                # Delete temporary JSON from S3
                s3.delete_object(Bucket=bucket_name, Key=latest_file_key)
                st.success("🧹 Temporary file deleted from S3.")

                st.success("🎉 Ingestion completed and log saved to S3!")

            except Exception as e:
                st.error(f"❌ Ingestion failed: {e}")
else:
    st.warning("⚠️ No inventory JSON files found in S3 under the `copilot/` folder.")
