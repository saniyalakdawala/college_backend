import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_unstructured import UnstructuredLoader


# Load environment variables (not needed for HuggingFace local embeddings, but kept for consistency)
load_dotenv()

# --- Step 1: Load the PDF Document (OCR if needed) ---
def load_pdf_with_ocr(file_path, use_ocr=True, poppler_path=None):
    """
    Loads documents from a file. Uses OCR only if specified.
    Automatically handles scanned PDFs.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # Determine strategy
    strategy = "ocr_only" if use_ocr else "auto"
    
    # Extra arguments for Poppler if using OCR
    extra_args = {"poppler_path": poppler_path} if poppler_path else None
    
    loader = UnstructuredLoader(
        file_path,
        mode="elements",
        strategy=strategy,
        languages=["eng"],
        extra_args=extra_args
    )
    
    try:
        docs = loader.load()
        text = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        raise RuntimeError(f"Failed to load PDF. Error: {e}")
    
    if text:
        print("Text extraction was successful!")
        print("--- First 2000 characters of the PDF ---")
        print(text[:2000])
        print("-------------------------")
    else:
        print("No text could be extracted. The document may be empty or corrupted.")
        
    return text

# --- Step 2: Chunk the Text ---
def split_text_into_chunks(text):
    if not text:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

# --- Step 3: Create Embeddings and Store in Vector DB ---
def create_embeddings_and_store(chunks, db_directory="./chroma_db"):
    if not chunks:
        print("No text chunks to embed. Database will be empty.")
        return
    
    print("Initializing embeddings model... (BAAI/bge-large-en-v1.5)")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"
    )

    print("Creating and saving vector store...")
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=db_directory
    )
    
    print("Vector database created successfully!")

# --- Main Execution ---
if __name__ == "__main__":
    pdf_path = "SRB_MPSTME_2024-25_sswFVjxQKq.pdf"
    
    # Optional: specify Poppler path if OCR is needed on Windows
    poppler_path = r"C:\Users\saniy\Release-25.07.0-0\poppler-25.07.0\Library\bin"  # Change if installed elsewhere
    
    try:
        raw_text = load_pdf_with_ocr(pdf_path, use_ocr=True, poppler_path=poppler_path)
        text_chunks = split_text_into_chunks(raw_text)
        print(f"Split into {len(text_chunks)} chunks.")
        create_embeddings_and_store(text_chunks)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please place your PDF file in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
