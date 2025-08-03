import streamlit as st
import os
import json
import tempfile
import time
from datetime import datetime
import boto3
from dotenv import load_dotenv
from google.cloud import asset_v1
from google.protobuf.json_format import MessageToDict
from pinecone import Pinecone, ServerlessSpec
from ingestion_script import LangchainPineconeLoader
from openai import OpenAI
import openai

# --------------- Setup ----------------
st.set_page_config(page_title="GCP Copilot + QnA", layout="wide")
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not set.")

bucket_name = "datacrux-dev"
log_key = "copilot/ingestion_log.json"
prefix = "copilot/"
s3 = boto3.client("s3")
pc = Pinecone(api_key=pinecone_api_key)
openai_client = OpenAI()

# ------------------------ #
# Utility Functions
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

def get_latest_index_from_s3_log():
    try:
        tmp_log_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
        s3.download_file(bucket_name, log_key, tmp_log_file)
        with open(tmp_log_file, "r") as f:
            ingestion_log = json.load(f)
        if not ingestion_log:
            raise ValueError("Ingestion log is empty.")
        latest_file = sorted(ingestion_log.keys(), reverse=True)[0]
        return ingestion_log[latest_file]
    except Exception as e:
        st.error(f"Error loading latest index from S3: {e}")
        return None

# ------------------------ #
# Sidebar Navigation
# ------------------------ #
page = st.sidebar.selectbox("Navigate", ["🔁 GCP Inventory + Ingestion", "🤖 Ask Questions (QnA)"])

# ------------------------ #
# Page 1: Inventory + Ingestion
# ------------------------ #
if page == "🔁 GCP Inventory + Ingestion":
    st.title(" GCP KloudCue AI Smart Inventory Assimilator")
    ingestion_log = load_ingestion_log()

    uploaded_key = st.file_uploader("📤 Upload GCP Service Account JSON", type="json")
    project_id = st.text_input(" Enter GCP Project ID")

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

                with open(local_path, 'w') as f:
                    json.dump(assets_json, f, indent=4)

                with open(local_path, 'rb') as f:
                    st.download_button("📥 Download GCP Inventory JSON", data=f, file_name=filename, mime="application/json")

                s3_key = f"{prefix}{filename}"
                s3.upload_file(local_path, bucket_name, s3_key)
                st.success(f"✅ Uploaded to `s3://{bucket_name}/{s3_key}`.")

            except Exception as e:
                st.error(f"❌ Failed to fetch/upload: {e}")

    st.markdown("---")
    st.subheader("  KloudCue AI Ingestion Orchestrator")

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
        st.write(f"📄 Latest JSON: `{latest_file_name}`")

        if latest_file_name in ingestion_log:
            st.info(f"✅ Already ingested into: `{ingestion_log[latest_file_name]}`")
        else:
            index_name = st.text_input("📇 New Pinecone Index Name")
            if index_name and st.button("📥 Start Ingestion"):
                try:
                    latest_local = os.path.join(tempfile.gettempdir(), latest_file_name)
                    s3.download_file(bucket_name, latest_file_key, latest_local)

                    loader = LangchainPineconeLoader(
                        bucket_name=bucket_name,
                        directory_path=prefix,
                        index_name=index_name
                    )
                    loader.load_and_index()

                    ingestion_log[latest_file_name] = index_name
                    save_ingestion_log(ingestion_log)
                    s3.delete_object(Bucket=bucket_name, Key=latest_file_key)
                    st.success("🎉 Ingestion completed!")
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")
    else:
        st.warning("Ingest JSON Inventory into Pinecone.")

# ------------------------ #
# Page 2: QnA Assistant
# ------------------------ #
elif page == "🤖 Ask Questions (QnA)":
    st.markdown("<h1 class='main-title'>GCP-GPT</h1>", unsafe_allow_html=True)
    st.markdown("<p class='description'>Ask questions about your GCP infrastructure — answers are based on latest ingested JSON from Pinecone</p>", unsafe_allow_html=True)

    index_name = get_latest_index_from_s3_log()
    if not index_name:
        st.stop()

    dimension = 1536
    metric = 'euclidean'
    spec = ServerlessSpec(cloud='aws', region='us-east-1')

    if index_name not in pc.list_indexes().names():
        pc.create_index(index_name, dimension=dimension, metric=metric, spec=spec)

    class PenTestVAPTAssistant:
        def __init__(self, index_name):
            self.index = pc.Index(index_name)
            self.embeddings_model = "text-embedding-3-small"
            self.llm_model = "gpt-4o-mini"

        def generate_embedding(self, text):
            response = openai_client.embeddings.create(input=text, model=self.embeddings_model)
            return response.data[0].embedding

        def search_index(self, query, top_k=6):
            vector = self.generate_embedding(query)
            return self.index.query(vector=vector, top_k=top_k, include_metadata=True)

        def retrieve_documents(self, results, max_docs=3):
            return [match['metadata']['content'] for match in results.get('matches', [])[:max_docs]]

        def generate_report(self, query, docs):
            prompt = f"Question: {query}\n\nRelevant Documents:\n" + "\n".join(f"- {d}" for d in docs)
            prompt += "\nProvide a detailed answer with references."
            role = "= Your Role =\nYou are a GCP JSON assistant..."
            messages = [{"role": "system", "content": role}, {"role": "user", "content": prompt}]
            response = openai_client.chat.completions.create(model=self.llm_model, messages=messages)
            return response.choices[0].message.content.strip()

        def query(self, question):
            results = self.search_index(question)
            docs = self.retrieve_documents(results)
            return self.generate_report(question, docs) if docs else "No relevant data found."

    st.sidebar.success(f"✅ Using Pinecone Index: `{index_name}`")

    if 'history' not in st.session_state:
        st.session_state.history = []

    with st.form("qna_form"):
        user_question = st.text_input("Ask a question:")
        submitted = st.form_submit_button("Ask")

    if submitted and user_question:
        assistant = PenTestVAPTAssistant(index_name)
        answer = assistant.query(user_question)
        st.markdown("###  Answer")
        st.markdown(answer)
        st.session_state.history.append((user_question, answer))

    if st.session_state.history:
        st.sidebar.write("### 📜 Previous Questions")
        for q, a in st.session_state.history:
            if st.sidebar.button(q):
                st.markdown("### 📌 Answer")
                st.markdown(a)
