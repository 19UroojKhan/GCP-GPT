# import streamlit as st
# import os
# import json
# import tempfile
# import time
# from datetime import datetime
# import boto3
# from dotenv import load_dotenv
# from google.cloud import asset_v1
# from google.protobuf.json_format import MessageToDict
# from pinecone import Pinecone, ServerlessSpec
# from ingestion_script import LangchainPineconeLoader
# from openai import OpenAI
# import openai

# # --------------- Setup ----------------
# st.set_page_config(page_title="GCP KloudCuee Copilot", layout="wide")
# load_dotenv()

# openai.api_key = os.getenv("OPENAI_API_KEY")
# pinecone_api_key = os.getenv("PINECONE_API_KEY")
# if not pinecone_api_key:
#     raise ValueError("PINECONE_API_KEY not set.")

# bucket_name = "datacrux-dev"
# log_key = "copilot/ingestion_log.json"
# prefix = "copilot/"
# s3 = boto3.client("s3")
# pc = Pinecone(api_key=pinecone_api_key)
# openai_client = OpenAI()

# # ------------------------ #
# # Utility Functions
# # ------------------------ #
# def load_ingestion_log():
#     try:
#         with tempfile.NamedTemporaryFile(delete=False) as tmp:
#             s3.download_file(bucket_name, log_key, tmp.name)
#             with open(tmp.name, "r") as f:
#                 return json.load(f)
#     except s3.exceptions.ClientError:
#         return {}

# def save_ingestion_log(log_data):
#     with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as tmp:
#         json.dump(log_data, tmp, indent=2)
#         tmp.flush()
#         s3.upload_file(tmp.name, bucket_name, log_key)

# def get_latest_index_from_s3_log():
#     try:
#         tmp_log_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
#         s3.download_file(bucket_name, log_key, tmp_log_file)
#         with open(tmp_log_file, "r") as f:
#             ingestion_log = json.load(f)
#         if not ingestion_log:
#             raise ValueError("Ingestion log is empty.")
#         latest_file = sorted(ingestion_log.keys(), reverse=True)[0]
#         return ingestion_log[latest_file]
#     except Exception as e:
#         st.error(f"Error loading latest index from S3: {e}")
#         return None

# # ------------------------ #
# # Sidebar Navigation
# # ------------------------ #
# page = st.sidebar.selectbox("Navigate", ["🔁 GCP Inventory + Ingestion", "🤖 Ask Questions (QnA)"])

# # ------------------------ #
# # Page 1: Inventory + Ingestion
# # ------------------------ #
# if page == "🔁 GCP Inventory + Ingestion":
#     st.title(" GCP KloudCue AI Smart Inventory Assimilator")
#     ingestion_log = load_ingestion_log()

#     uploaded_key = st.file_uploader("📤 Upload GCP Service Account JSON", type="json")
#     project_id = st.text_input(" Enter GCP Project ID")

#     if uploaded_key and project_id:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
#             tmp.write(uploaded_key.read())
#             service_key_path = tmp.name

#         os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_key_path
#         st.success("✅ Service account saved and configured")

#         if st.button("🚀 Fetch GCP Asset Inventory"):
#             st.write("🔄 Fetching assets from GCP...")

#             def list_assets(project_id):
#                 client = asset_v1.AssetServiceClient()
#                 parent = f"projects/{project_id}"
#                 request = {"parent": parent, "content_type": asset_v1.ContentType.RESOURCE}
#                 assets = {"assets": []}
#                 for asset in client.list_assets(request=request):
#                     assets["assets"].append(MessageToDict(asset._pb))
#                 return assets

#             try:
#                 assets_json = list_assets(project_id)
#                 timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#                 filename = f"gcp_inventory_{timestamp}.json"
#                 local_path = os.path.join(tempfile.gettempdir(), filename)

#                 with open(local_path, 'w') as f:
#                     json.dump(assets_json, f, indent=4)

#                 with open(local_path, 'rb') as f:
#                     st.download_button("📥 Download GCP Inventory JSON", data=f, file_name=filename, mime="application/json")

#                 s3_key = f"{prefix}{filename}"
#                 s3.upload_file(local_path, bucket_name, s3_key)
#                 st.success(f"✅ Uploaded to `s3://{bucket_name}/{s3_key}`.")

#             except Exception as e:
#                 st.error(f"❌ Failed to fetch/upload: {e}")

#     st.markdown("---")
#     st.subheader("  KloudCue AI Ingestion Orchestrator")

#     try:
#         objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
#         json_files = sorted(
#             [obj for obj in objects.get("Contents", []) if obj["Key"].endswith(".json") and "ingestion_log" not in obj["Key"]],
#             key=lambda x: x["LastModified"],
#             reverse=True
#         )
#     except Exception as e:
#         st.error(f"❌ Failed to list S3 objects: {e}")
#         json_files = []

#     if json_files:
#         latest_file_key = json_files[0]["Key"]
#         latest_file_name = os.path.basename(latest_file_key)
#         st.write(f"📄 Latest JSON: `{latest_file_name}`")

#         if latest_file_name in ingestion_log:
#             st.info(f"✅ Already ingested into: `{ingestion_log[latest_file_name]}`")
#         else:
#             index_name = st.text_input("📇 New Pinecone Index Name")
#             if index_name and st.button("📥 Start Ingestion"):
#                 try:
#                     latest_local = os.path.join(tempfile.gettempdir(), latest_file_name)
#                     s3.download_file(bucket_name, latest_file_key, latest_local)

#                     loader = LangchainPineconeLoader(
#                         bucket_name=bucket_name,
#                         directory_path=prefix,
#                         index_name=index_name
#                     )
#                     loader.load_and_index()

#                     ingestion_log[latest_file_name] = index_name
#                     save_ingestion_log(ingestion_log)
#                     s3.delete_object(Bucket=bucket_name, Key=latest_file_key)
#                     st.success("🎉 Ingestion completed!")
#                 except Exception as e:
#                     st.error(f"❌ Ingestion failed: {e}")
#     else:
#         st.warning("Ingest JSON Inventory into Pinecone.")

# # ------------------------ #
# # Page 2: QnA Assistant
# # ------------------------ #
# elif page == "🤖 Ask Questions (QnA)":
#     st.markdown("<h1 class='main-title'>GCP-GPT</h1>", unsafe_allow_html=True)
#     st.markdown("<p class='description'>Ask questions about your GCP infrastructure — answers are based on latest ingested JSON from Pinecone</p>", unsafe_allow_html=True)

#     index_name = get_latest_index_from_s3_log()
#     if not index_name:
#         st.stop()

#     dimension = 1536
#     metric = 'euclidean'
#     spec = ServerlessSpec(cloud='aws', region='us-east-1')

#     if index_name not in pc.list_indexes().names():
#         pc.create_index(index_name, dimension=dimension, metric=metric, spec=spec)

#     class PenTestVAPTAssistant:
#         def __init__(self, index_name):
#             self.index = pc.Index(index_name)
#             self.embeddings_model = "text-embedding-3-small"
#             self.llm_model = "gpt-4o-mini"

#         def generate_embedding(self, text):
#             response = openai_client.embeddings.create(input=text, model=self.embeddings_model)
#             return response.data[0].embedding

#         def search_index(self, query, top_k=6):
#             vector = self.generate_embedding(query)
#             return self.index.query(vector=vector, top_k=top_k, include_metadata=True)

#         def retrieve_documents(self, results, max_docs=3):
#             return [match['metadata']['content'] for match in results.get('matches', [])[:max_docs]]

#         def generate_report(self, query, docs):
#             prompt = f"Question: {query}\n\nRelevant Documents:\n" + "\n".join(f"- {d}" for d in docs)
#             prompt += "\nProvide a detailed answer with references."
#             role = "= Your Role =\nYou are a GCP JSON assistant..."
#             messages = [{"role": "system", "content": role}, {"role": "user", "content": prompt}]
#             response = openai_client.chat.completions.create(model=self.llm_model, messages=messages)
#             return response.choices[0].message.content.strip()

#         def query(self, question):
#             results = self.search_index(question)
#             docs = self.retrieve_documents(results)
#             return self.generate_report(question, docs) if docs else "No relevant data found."

#     st.sidebar.success(f"✅ Using Pinecone Index: `{index_name}`")

#     if 'history' not in st.session_state:
#         st.session_state.history = []

#     with st.form("qna_form"):
#         user_question = st.text_input("Ask a question:")
#         submitted = st.form_submit_button("Ask")

#     if submitted and user_question:
#         assistant = PenTestVAPTAssistant(index_name)
#         answer = assistant.query(user_question)
#         st.markdown("###  Answer")
#         st.markdown(answer)
#         st.session_state.history.append((user_question, answer))

#     if st.session_state.history:
#         st.sidebar.write("### Chat History")
#         for q, a in st.session_state.history:
#             if st.sidebar.button(q):
#                 st.markdown("### 📌 Answer")
#                 st.markdown(a)


import streamlit as st
import os
import json
import tempfile
import re
import logging
from datetime import datetime
from typing import Dict, Optional

import boto3
from dotenv import load_dotenv

from google.cloud import asset_v1
from google.protobuf.json_format import MessageToDict
from pinecone import Pinecone, ServerlessSpec
from ingestion_script import LangchainPineconeLoader
from openai import OpenAI
import openai

# ------------------------ #
# Configuration & Logging
# ------------------------ #
load_dotenv()
LOG = logging.getLogger("gcp_copilot")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
LOG.addHandler(handler)

# Streamlit page config
st.set_page_config(page_title="GCP Copilot + QnA", layout="wide")

# Required environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
S3_BUCKET = os.getenv("S3_BUCKET", "datacrux-dev")
INGESTION_LOG_KEY = os.getenv("INGESTION_LOG_KEY", "copilot/ingestion_log.json")
S3_PREFIX = os.getenv("S3_PREFIX", "copilot/")

# Validate essential env vars early and fail fast with helpful message
if not OPENAI_API_KEY:
    st.error("Environment variable OPENAI_API_KEY is not set. Please add it to your .env or CI environment.")
    st.stop()
if not PINECONE_API_KEY:
    st.error("Environment variable PINECONE_API_KEY is not set. Please add it to your .env or CI environment.")
    st.stop()

openai.api_key = OPENAI_API_KEY
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize AWS S3 client
s3 = boto3.client("s3")

# Initialize Pinecone client wrapper
pc = Pinecone(api_key=PINECONE_API_KEY)

# ------------------------ #
# Helper utilities
# ------------------------ #

def sanitize_index_name(name: str) -> str:
    """Make a Pinecone-safe index name.
    Replaces invalid characters with '-' and trims repeating hyphens.
    """
    name = re.sub(r"[^a-z0-9\-]", "-", name.lower())
    name = re.sub(r"-+", "-", name)
    return name.strip("-")


def load_ingestion_log(bucket: str = S3_BUCKET, key: str = INGESTION_LOG_KEY) -> Dict:
    """Download and parse the ingestion log from S3.
    Returns an empty dict if log doesn't exist.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            s3.download_file(bucket, key, tmp.name)
            with open(tmp.name, "r") as f:
                data = json.load(f)
                LOG.info("Loaded ingestion log from S3: %s", key)
                return data
    except Exception as e:
        LOG.warning("Could not load ingestion log from s3://%s/%s -> %s", bucket, key, e)
        return {}


def save_ingestion_log(log_data: Dict, bucket: str = S3_BUCKET, key: str = INGESTION_LOG_KEY) -> None:
    """Save ingestion log into S3 (atomic write via temp file)."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as tmp:
            json.dump(log_data, tmp, indent=2)
            tmp.flush()
            s3.upload_file(tmp.name, bucket, key)
        LOG.info("Saved ingestion log to s3://%s/%s", bucket, key)
    except Exception as e:
        LOG.exception("Failed to save ingestion log: %s", e)
        raise


def get_latest_index_from_log(log: Dict) -> Optional[str]:
    """Return the latest created Pinecone index from ingestion log (value of the most recent key)."""
    if not log:
        return None
    # ingestion log expected to be {filename: index_name, ...} with timestamps in filename
    # choose the latest by file modification-like ordering (sorted by key if follow naming convention)
    try:
        latest_filename = sorted(log.keys())[-1]
        return log[latest_filename]
    except Exception:
        return None


def pinecone_index_exists(index_name: str) -> bool:
    try:
        indexes = pc.list_indexes()
        # pc.list_indexes() may return a list or an object; normalize
        if isinstance(indexes, (list, tuple)):
            return index_name in indexes
        # otherwise attempt to iterate
        names = [i.name if hasattr(i, 'name') else i for i in indexes]
        return index_name in names
    except Exception:
        LOG.exception("Failed to list Pinecone indexes")
        return False


# ------------------------ #
# Page: Sidebar Navigation
# ------------------------ #

page = st.sidebar.selectbox("CatalystOps Navigation", [
    "🔁 GCP Inventory + Ingestion",
    "🤖 AI Assistant (QnA)",
    "🛠️ Generate Terraform Code",
])

# Persist simple state
if 'history' not in st.session_state:
    st.session_state.history = []

# ------------------------ #
# Page: Inventory + Ingestion
# ------------------------ #
if page == "🔁 GCP Inventory + Ingestion":
    st.title("GCP CatalystOps AutoPilot")
    st.markdown("""
 **CatalystOps** helps you **discover, manage, and provision GCP resources effortlessly**.  
 Ingest your cloud inventory, get **AI-powered insights**, and generate **ready-to-generate Terraform code** ~ all in one place.  
""")

    ingestion_log = load_ingestion_log()

    uploaded_key = st.file_uploader("📤 Connect to GCP Account", type="json")
    extra_prefix = st.text_input("S3 prefix to upload inventory to:", value=S3_PREFIX)

    if uploaded_key:
        # Save service account locally
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(uploaded_key.read())
            service_key_path = tmp.name

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_key_path
        st.success("✅ Connected to GCP Account successfully")

        # Extract project_id from service account file
        with open(service_key_path, "r") as f:
            service_account_data = json.load(f)
        project_id = service_account_data.get("project_id")
        st.success(f"✅ Service account saved — Project: `{project_id}`")

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
            # 1. Fetch assets
            assets_json = list_assets(project_id)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"gcp_inventory_{timestamp}.json"
            local_path = os.path.join(tempfile.gettempdir(), filename)

            # 2. Save locally
            with open(local_path, 'w') as f:
                json.dump(assets_json, f, indent=4)
            st.success(f"💾 Downloaded JSON locally: `{local_path}`")

            # 3. Push to S3
            s3_key = f"{extra_prefix.rstrip('/')}/{filename}"
            s3.upload_file(local_path, S3_BUCKET, s3_key)
            st.success(f"☁️ Pushed to `s3://{S3_BUCKET}/{s3_key}` for ingestion")

            # 4. Create sanitized Pinecone index name
            auto_index_name = sanitize_index_name(os.path.splitext(filename)[0])
            st.info(f"📇 Index to be created: `{auto_index_name}`")

            # 5. Create Pinecone index if not exists
            if not pinecone_index_exists(auto_index_name):
                st.write("📌 Creating Pinecone index...")
                pc.create_index(
                    auto_index_name,
                    dimension=1536,
                    metric='euclidean',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                st.success(f"✅ Index created: `{auto_index_name}`")
            else:
                st.info(f"Index `{auto_index_name}` already exists — will reuse it for ingestion.")

            # 6. Ingest data using user's loader
            st.write("🚚 Ingestion started — Processing...")
            loader = LangchainPineconeLoader(
                bucket_name=S3_BUCKET,
                directory_path=extra_prefix.rstrip('/') + '/',
                index_name=auto_index_name
            )
            loader.load_and_index()

            # 7. Update ingestion log
            ingestion_log[filename] = auto_index_name
            save_ingestion_log(ingestion_log)

            # 8. Optionally cleanup uploaded file in S3
            if st.checkbox("Delete inventory file from S3 after ingestion", value=True):
                s3.delete_object(Bucket=S3_BUCKET, Key=s3_key)
                st.info("Uploaded inventory file deleted from S3.")

            st.success(f"🎉 Ingestion completed into `{auto_index_name}`!")

        except Exception as e:
            LOG.exception("Ingestion pipeline failed")
            st.error(f"❌ Process failed: {e}")


# ------------------------ #
# Page: QnA Assistant
# ------------------------ #
elif page == "🤖 AI Assistant (QnA)":
    st.markdown("<h1 class='main-title'>CatalystOps AI Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p class='description'>Ask questions about your GCP infrastructure ~ From data to decisions, in seconds. </p>", unsafe_allow_html=True)

    ingestion_log = load_ingestion_log()
    index_name = get_latest_index_from_log(ingestion_log := ingestion_log) if "get_latest_index_from_log" in globals() else get_latest_index_from_log(ingestion_log)
    # fallback: a simple helper
    index_name = get_latest_index_from_log(ingestion_log)

    if not index_name:
        st.warning("No Pinecone index found in ingestion log. Run an ingestion first.")
        st.stop()

    # Ensure index exists
    if not pinecone_index_exists(index_name):
        try:
            pc.create_index(index_name, dimension=1536, metric='euclidean', spec=ServerlessSpec(cloud='aws', region='us-east-1'))
            st.info(f"Created missing index `{index_name}`")
        except Exception as e:
            st.error(f"Failed to create or connect to Pinecone index `{index_name}`: {e}")
            st.stop()

    class GCPJsonAssistant:
        def __init__(self, index_name: str):
            self.index = pc.Index(index_name)
            self.embeddings_model = "text-embedding-3-small"
            self.llm_model = "gpt-4o-mini"

        def generate_embedding(self, text: str):
            # small helper to consistently create embeddings and handle rate limits/errors
            try:
                response = openai_client.embeddings.create(input=text, model=self.embeddings_model)
                return response.data[0].embedding
            except Exception:
                LOG.exception("Failed to create embedding")
                raise

        def search_index(self, query: str, top_k: int = 6):
            vector = self.generate_embedding(query)
            return self.index.query(vector=vector, top_k=top_k, include_metadata=True)

        def retrieve_documents(self, results, max_docs: int = 3):
            return [match.get('metadata', {}).get('content', '') for match in results.get('matches', [])[:max_docs]]

        def generate_report(self, query: str, docs):
            prompt = f"Question: {query}\n\nRelevant Documents:\n" + "\n".join(f"- {d}" for d in docs)
            prompt += "\nProvide a detailed answer with references."
            role = "= Your Role =\nYou are a GCP JSON assistant..."
            messages = [{"role": "system", "content": role}, {"role": "user", "content": prompt}]
            try:
                response = openai_client.chat.completions.create(model=self.llm_model, messages=messages)
                return response.choices[0].message.content.strip()
            except Exception:
                LOG.exception("LLM call failed")
                return "LLM request failed. Check logs."

        def query(self, question: str):
            results = self.search_index(question)
            docs = self.retrieve_documents(results)
            return self.generate_report(question, docs) if docs and any(docs) else "No relevant data found."

    st.sidebar.success(f"✅ Using Pinecone Index: `{index_name}`")

    with st.form("qna_form"):
        user_question = st.text_input("Ask a question:")
        submitted = st.form_submit_button("Ask")

    if submitted and user_question:
        assistant = GCPJsonAssistant(index_name)
        with st.spinner("Thinking..."):
            try:
                answer = assistant.query(user_question)
                st.markdown("### Answer")
                st.markdown(answer)
                st.session_state.history.append((user_question, answer))
            except Exception as e:
                st.error(f"Error while querying assistant: {e}")

    if st.session_state.history:
        st.sidebar.write("### 📜 Previous Questions")
        import hashlib
        for q, a in reversed(st.session_state.history[-20:]):
            q_hash = hashlib.md5(q.encode()).hexdigest()
            if st.sidebar.button(q, key=f"question_btn_{q_hash}"):
                st.markdown("### 📌 Answer")
                st.markdown(a)

        # for q, a in reversed(st.session_state.history[-20:]):
        #     if st.sidebar.button(q):
        #         st.markdown("### 📌 Answer")
        #         st.markdown(a)


# ------------------------ #
# Page: Terraform Code Generator
# ------------------------ #

elif page == "🛠️ Generate Terraform Code":
    st.title(" GCP Terraform Code Generator")
    st.markdown("Generate Terraform IaC snippets with embedded best practices.")

    terraform_prompt = st.text_area("What infrastructure do you want to generate Terraform for?")

    class TerraformAssistant:
        def __init__(self):
            self.llm_model = "gpt-4.1"

        def generate_terraform_code(self, user_request):
            prompt = f"""
Generate Terraform code for the following GCP infrastructure request.

=== USER REQUEST ===
{user_request}

Constraints:
- Output the following files:
  - main.tf
  - variables.tf
  - outputs.tf
  - backend.tf
  - provider.tf
  - terraform.tfvars
- Follow industry best practices for Terraform, including:
  - Use of modules for reusable components
  - Separate state files for dev and prod environments
  - Remote backend (e.g., GCS bucket) usage for state management
  - Resource tagging with owner and product
  - RBAC/IAM policies configured securely
  - Use variables and outputs for customization and reusability
  - Use of lock files and version constraints
  - Proper indentation and formatting
- Generate valid, production-grade Terraform code.
- Do not include explanations or comments. Only output the code.
"""
            messages = [
                {"role": "system", "content": "You are a Terraform expert for GCP. Generate production-grade standalone Terraform code following best practices."},
                {"role": "user", "content": prompt}
            ]
            response = openai_client.chat.completions.create(model=self.llm_model, messages=messages)
            return response.choices[0].message.content.strip()

    if st.button(" Generate Terraform Code..."):
        if not terraform_prompt.strip():
            st.warning("Please describe what you want to generate.")
        else:
            with st.spinner("Generating Terraform code..."):
                try:
                    assistant = TerraformAssistant()
                    terraform_code = assistant.generate_terraform_code(terraform_prompt)
                    st.code(terraform_code, language="hcl")
                    # st.download_button("💾 Download main.tf", data=terraform_code, file_name="main.tf", mime="text/plain")
                    st.success("Terraform code generated with best practices.")
                except Exception as e:
                    st.error(f"❌ Error generating Terraform code: {e}")


