import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the Gemini API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# --- Initialize RAG components ---
# IMPORTANT: This must match the model used in ingest_data.py
print("Initializing embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

print("Loading persistent vector database...")
try:
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    num_docs = db._collection.count()
    print(f"ChromaDB contains {num_docs} documents.")
except Exception as e:
    print(f"Error accessing ChromaDB: {e}. Please run ingest_data.py first.")
    db = None

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
) if db else None

# Initialize LLM for streaming
llm_stream = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-001",  # Corrected model name for streaming
    google_api_key=api_key,
    temperature=0.2,
    streaming=True # Ensure streaming is enabled
)

# Define a custom prompt template for the RAG chain
template = """You are a helpful assistant. Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
CONTEXT: {context}
----------------
QUESTION: {question}
----------------
ANSWER:"""
rag_prompt = PromptTemplate(template=template, input_variables=["context", "question"])

print("Backend is ready to receive queries.")

# --- Greeting handler ---
def handle_greetings(query):
    greetings = ["hi", "hello", "hey", "what's up", "yo"]
    if query.lower().strip() in greetings:
        return "Hello! How can I help you today?"
    return None

# --- API Endpoint for streaming ---
@app.route("/search_and_stream", methods=["POST"])
def search_and_stream():
    data = request.json
    user_query = data.get("query", "")

    if not retriever:
        return jsonify({"answer": "Error: Vector database not loaded."}), 500

    greeting_response = handle_greetings(user_query)
    if greeting_response:
        return jsonify({"answer": greeting_response})

    if not user_query:
        return jsonify({"answer": "Please provide a query."}), 400

    try:
        # Step 1: Retrieve documents first
        retrieved_docs = retriever.get_relevant_documents(user_query)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Step 2: Create a prompt with the context
        prompt_with_context = rag_prompt.format(context=context_text, question=user_query)

        # Step 3: Stream the response from the LLM
        def generate():
            for chunk in llm_stream.stream(prompt_with_context):
                yield chunk.content or "" # Yield each content chunk

        return Response(stream_with_context(generate()), mimetype='text/plain')

    except Exception as e:
        print(f"Error processing streaming request: {e}")
        return jsonify({"answer": "An error occurred while generating a response. Please try again later."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
