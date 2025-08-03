


import streamlit as st
import os
import json
import tempfile
import time
from dotenv import load_dotenv
import openai
import boto3
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is not set in environment variables.")

pc = Pinecone(api_key=pinecone_api_key)


# 🔁 Step 1: Get Latest Ingested Index Name from S3 Log
def get_latest_index_from_s3_log(bucket_name="datacrux-dev", log_key="copilot/ingestion_log.json"):
    try:
        s3 = boto3.client("s3")
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

# ✅ Get index name dynamically
index_name = get_latest_index_from_s3_log()
if not index_name:
    st.stop()

dimension = 1536
metric = 'euclidean'
spec = ServerlessSpec(cloud='aws', region='us-east-1')

# Create index if it doesn't exist
def create_index_with_retry(index_name, dimension, metric, spec, retries=3, delay=5):
    for attempt in range(retries):
        try:
            if index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=spec
                )
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                st.error(f"Failed to create index after {retries} attempts: {e}")
                return False

create_index_with_retry(index_name, dimension, metric, spec)

MODEL = "gpt-4o-mini"

class PenTestVAPTAssistant:
    def __init__(self, index_name, embeddings_model="text-embedding-3-small", llm_model=MODEL):
        self.openai = openai
        self.pinecone = pc
        self.index = self.pinecone.Index(index_name)
        self.embeddings_model = embeddings_model
        self.llm_model = llm_model
        self.client = OpenAI()

    def generate_embedding(self, text):
        try:
            response = self.client.embeddings.create(input=text, model=self.embeddings_model)
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Error generating embedding: {e}")
            return None

    def search_index(self, query, top_k=6):
        embedding = self.generate_embedding(query)
        if embedding is None:
            return None
        try:
            return self.index.query(
                vector=embedding,
                top_k=top_k,
                include_values=False,
                include_metadata=True
            )
        except Exception as e:
            st.error(f"Error querying index: {e}")
            return None

    def retrieve_documents(self, query_result, max_docs=3):
        documents = []
        if not query_result or 'matches' not in query_result:
            return documents
        for result in query_result['matches'][:max_docs]:
            try:
                documents.append(result['metadata']['content'])
            except KeyError:
                st.error(f"Document ID '{result['id']}' not found in metadata.")
        return documents

    def generate_report(self, query, documents):
        prompt = f"Question: {query}\n\nRelevant Documents:\n"
        for doc in documents:
            prompt += f"- {doc}\n"
        # prompt += "\nProvide a detailed answer to the question above based on the relevant documents. Include references in the format 'Reference: [source information]'."

        # role_description = (
        #     "= Your Role =\n"
        #     "Your role is to act as a GCP Assistant and answer queries strictly based on structured JSON data.\n"
        #     "Do not hallucinate or use external knowledge. Use only the ingested Pinecone data.\n\n"
        #     "Example Queries:\n"
        #     "- How many buckets exist?\n"
        #     "- Show VMs without encryption\n"
        #     "- Buckets with lifecycle rules\n"
        # )
        prompt += "\nProvide a detailed answer to the question above based on the relevant documents. Include references in the format 'Reference: [source information]'." 

        role_description = (
    "= Your Role =\n"
    "Your primary role is to act as a technical assistant capable of answering detailed queries about Google Cloud Platform (GCP) infrastructure. "
    "You do this by analyzing and retrieving accurate information from a structured JSON dataset that represents the current state of GCP resources (buckets, instances, firewalls, etc.) indexed in Pinecone.\n\n"

    "= Your Knowledge Base =\n"
    "- Your knowledge base consists exclusively of a structured JSON dump of GCP infrastructure resources ingested into Pinecone.\n"
    "- You do not use general knowledge or assumptions about GCP. You rely strictly on the actual ingested dataset.\n"
    "- You understand the JSON structure and the fields of each GCP resource type (such as `encryption`, `lifecycle`, `status`, `name`, etc.).\n\n"

    "= Your Job =\n"
    "Upon receiving a user query related to GCP resources (e.g., number of buckets, instances without encryption, existence of lifecycle rules), "
    "you will:\n"
    "- Search and extract relevant data from the Pinecone-indexed JSON documents.\n"
    "- Interpret the structure to deliver precise, fact-based answers.\n"
    "- Present the results clearly with supporting logic and references to the relevant data points.\n"
    "- If the query is unclear or missing specifics (like project ID, resource type, etc.), ask the user for clarification.\n\n"

    "= How to Work with User Queries =\n"
    "1. Clarify ambiguous or incomplete questions by requesting specific details from the user.\n"
    "2. Parse the question to determine the intent (e.g., count, configuration, compliance check).\n"
    "3. Search the ingested Pinecone documents to extract matching JSON data.\n"
    "4. Analyze and summarize the relevant portions with supporting explanations.\n"
    "5. Present the results in structured, clear, and concise language. Use tables or bullet points when helpful.\n"
    "6. Do not hallucinate or assume missing data — only respond based on what is found in the indexed dataset.\n"
    "7. If data is missing or incomplete, state that clearly.\n\n"

    "= Example Queries You Should Be Able to Handle =\n"
    "- \"How many buckets are configured in our GCP project?\"\n"
    "- \"List all VM instances without encryption enabled.\"\n"
    "- \"Do any S3 (Cloud Storage) buckets have lifecycle rules defined?\"\n"
    "- \"Which instances are using public IPs?\"\n"
    "- \"Show firewall rules that allow 0.0.0.0/0 ingress.\"\n\n"

    "= Outputs =\n"
    "Your output should include:\n"
    "- A direct answer to the question based on JSON data from Pinecone.\n"
    "- References to the matching JSON fields and values (e.g., `bucket.lifecycle`, `instance.encryption.enabled`, etc.).\n"
    "- Explanation of how the answer was derived.\n"
    "- Structured formatting when applicable (e.g., tables or bullet points).\n"
    "- Clarification questions if the query lacks context or specifics.\n"
)

        messages = [
            {"role": "system", "content": role_description},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages
            )
            report = response.choices[0].message.content.strip()
            references = self.extract_references(report)
            return report, references
        except Exception as e:
            st.error(f"An error occurred while generating the report: {e}")
            return None, []

    def extract_references(self, report):
        return [line for line in report.split("\n") if line.startswith("Reference:")]

    def query(self, query):
        query_result = self.search_index(query)
        if query_result is None:
            return None, []
        documents = self.retrieve_documents(query_result)
        if not documents:
            st.warning("No relevant documents found.")
            return None, []
        return self.generate_report(query, documents)

# Streamlit UI Setup
# st.set_page_config(page_title="GCP-GPT", layout="wide")

# Styling
st.markdown(f"""
    <style>
        body {{ font-family: 'Arial'; }}
        .main-title {{ font-size: 2.5rem; color: #000; text-align: center; margin-bottom: 25px; }}
        .description {{ font-size: 1.2rem; color: #333; text-align: center; margin-bottom: 50px; }}
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-title'>GCP-GPT</h1>", unsafe_allow_html=True)
st.markdown("<p class='description'>Ask questions about your GCP infrastructure — answers are based on latest ingested JSON from Pinecone.</p>", unsafe_allow_html=True)

# Show latest index info
st.sidebar.success(f"✅ Using Pinecone Index: `{index_name}`")

# Ask question
with st.form(key='question_form'):
    user_question = st.text_input("Enter your question here:", autocomplete='off')
    submit_button = st.form_submit_button(label='Ask')

# History state
if 'history' not in st.session_state:
    st.session_state.history = []

assistant = PenTestVAPTAssistant(index_name=index_name)

answer_placeholder = st.empty()
references_placeholder = st.empty()

if submit_button and user_question.strip():
    answer_placeholder.empty()
    references_placeholder.empty()

    report, references = assistant.query(user_question)
    if report:
        st.session_state.history.append({
            "question": user_question,
            "answer": report,
            "references": references
        })

        answer_placeholder.markdown("**Answer:**")
        for line in report.split("\n"):
            st.markdown(line)

        if references:
            references_placeholder.markdown("**References:**")
            for ref in references:
                references_placeholder.markdown(f"- {ref}")
    else:
        st.write("No response generated.")

# Show history
if st.session_state.history:
    st.sidebar.write("### Question History")
    for i, entry in enumerate(st.session_state.history):
        if st.sidebar.button(entry['question'], key=f"history_{i}"):
            answer_placeholder.markdown(f"**Answer:** {entry['answer']}")
            if entry['references']:
                references_placeholder.markdown("**References:**")
                for ref in entry['references']:
                    references_placeholder.markdown(f"- {ref}")

