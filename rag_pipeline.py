"""
Document RAG System using LangChain, Google Gemini, ChromaDB with STT -> LLM -> TTS
===================================================================================
Converted from RAG_LangChain.ipynb with integrated Voice Input (STT) and Voice Output (TTS).

Pipeline Steps:
  1. Dependencies & Environment
  2. Configure Gemini API Key (.env / environment)
  3. PDF Document Selection
  4. Document Loading (PyPDFLoader)
  5. Text Chunking (RecursiveCharacterTextSplitter)
  6. Embeddings & Vector Store (GoogleGenerativeAIEmbeddings & Chroma with rate-limit throttling)
  7. Retriever & LLM Configuration (ChatGoogleGenerativeAI)
  8. LCEL RAG Chain (Prompt | Retriever | LLM | StrOutputParser)
  9. STT -> LLM -> TTS Interactive Query Execution:
     - Voice STT: User asks question via microphone (with text fallback)
     - LLM RAG: Retrieves relevant chunks and generates answer
     - Text Output: Displays retrieved chunks (page & text) + final answer
     - TTS Output: Speaks ONLY the final answer out loud
"""

import os
import sys
import time
import re
from pathlib import Path
from dotenv import load_dotenv

# LangChain and Document Processing Imports
try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    PyPDFLoader = None

try:
    import pypdf
except ImportError:
    pypdf = None

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# STT & TTS dependencies
import speech_recognition as sr
import pyttsx3


# =====================================================================
# STT & TTS Helpers
# =====================================================================
def record_audio_via_sounddevice(sample_rate: int = 16000, max_duration: int = 10, silence_timeout: float = 1.5):
    """
    Captures microphone audio directly using sounddevice and returns sr.AudioData.
    Natively supports Windows without requiring PyAudio or C++ build tools.
    """
    import sounddevice as sd
    import numpy as np

    chunk_duration = 0.2
    chunk_samples = int(sample_rate * chunk_duration)
    chunks = []
    has_spoken = False
    silence_start = None

    print(f"\n[STT - Microphone] Listening (up to {max_duration}s)... Speak your question now.")
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
        start_time = time.time()
        while time.time() - start_time < max_duration:
            data, _ = stream.read(chunk_samples)
            chunks.append(data)
            energy = np.abs(data).mean()
            if energy > 250:
                has_spoken = True
                silence_start = None
            elif has_spoken:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > silence_timeout:
                    print("[STT] Silence detected, finishing recording.")
                    break

    if not chunks:
        return None

    audio_bytes = np.concatenate(chunks, axis=0).tobytes()
    return sr.AudioData(audio_bytes, sample_rate=sample_rate, sample_width=2)


def capture_speech_to_text(timeout: int = 8, phrase_time_limit: int = 15) -> str:
    """Captures microphone audio and transcribes it to text using SpeechRecognition."""
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.0

    audio = None

    # Primary method: sounddevice (no PyAudio required)
    try:
        audio = record_audio_via_sounddevice(max_duration=phrase_time_limit)
    except Exception as sd_err:
        # Fallback to standard PyAudio if available
        try:
            with sr.Microphone() as source:
                print("\n[STT] Adjusting for ambient noise... (1 sec)")
                recognizer.adjust_for_ambient_noise(source, duration=1.0)
                print(f"[STT] Speak your question now (limit {phrase_time_limit}s)...")
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except Exception:
            print(f"[STT Note] Microphone input unavailable: {sd_err}")

    if audio:
        try:
            print("[STT] Transcribing voice audio with Google STT...")
            query = recognizer.recognize_google(audio)
            print(f"[STT] Detected Question: \"{query}\"")
            return query.strip()
        except sr.UnknownValueError:
            print("[STT] Could not understand the voice audio. Please speak clearly.")
        except Exception as e:
            print(f"[STT Note] Speech recognition error: {e}")

    # Fallback to manual text input
    print("[STT Fallback] Please type your question manually:")
    return input(">> ").strip()


def speak_answer_tts(text: str):
    """Speaks ONLY the final answer using pyttsx3, stripping markdown formatting."""
    # Clean markdown formatting so speech sounds natural
    clean_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    clean_text = re.sub(r'https?://\S+', '', clean_text)
    clean_text = re.sub(r'[*_~#>`]', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    if not clean_text:
        return

    print("\n[TTS] Speaking final answer out loud...")
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        engine.setProperty("volume", 1.0)
        engine.say(clean_text)
        engine.runAndWait()
    except Exception as e:
        print(f"[TTS Note] Audio output could not play: {e}")


# =====================================================================
# Main Pipeline Runner
# =====================================================================
def run_pipeline():
    # Step 1 & 2: Environment & API Key Setup
    print("\n" + "=" * 70)
    print("  STEP 1 & 2: Initializing Environment & Gemini API Key")
    print("=" * 70)

    # Load from .env file
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv(".env")

    # Check environment for API key (supports single GEMINI_API_KEY)
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not api_key or api_key == "your_gemini_api_key_here":
        import getpass
        print("\n[!] No valid API key detected in .env or environment variables.")
        api_key = getpass.getpass("Enter your Google Gemini API Key: ").strip()

    if not api_key:
        print("[Error] Gemini API Key is required to run the RAG pipeline.")
        sys.exit(1)

    # Set only GOOGLE_API_KEY in os.environ to avoid duplicate key warnings from langchain-google-genai
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ.pop("GEMINI_API_KEY", None)
    print("✓ Gemini API key successfully configured for LangChain & Google GenAI!")

    # Check network connectivity to Google API endpoint
    import socket
    try:
        socket.getaddrinfo("generativelanguage.googleapis.com", 443)
    except socket.gaierror as e:
        print("\n" + "!" * 70)
        print(" [NETWORK ERROR] Cannot reach Google Generative AI (generativelanguage.googleapis.com)")
        print(f" Error detail: [Errno 11001] {e}")
        print(" Potential causes:")
        print("  1. No active internet connection on this computer.")
        print("  2. A corporate VPN or office firewall is blocking DNS or external APIs.")
        print("  3. If your network uses a proxy, configure it in PowerShell before running:")
        print("     $env:HTTP_PROXY = \"http://your-proxy-server:port\"")
        print("     $env:HTTPS_PROXY = \"http://your-proxy-server:port\"")
        print("!" * 70 + "\n")

    # Step 3: PDF Document Selection
    print("\n" + "=" * 70)
    print("  STEP 3: PDF Document Selection")
    print("=" * 70)

    # Check data/ and current directory
    search_locations = [Path("data"), Path(".")]
    pdf_files = []
    for loc in search_locations:
        if loc.exists():
            pdf_files.extend(list(loc.glob("*.pdf")))

    pdf_path = None
    if pdf_files:
        print(f"Found {len(pdf_files)} PDF file(s):")
        for idx, f in enumerate(pdf_files, 1):
            print(f"  [{idx}] {f}")
        choice = input(f"Select PDF [1-{len(pdf_files)}] or press Enter for default [{pdf_files[0].name}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(pdf_files):
            pdf_path = str(pdf_files[int(choice) - 1])
        else:
            pdf_path = str(pdf_files[0])
        print(f"Using PDF: {pdf_path}")
    else:
        pdf_input = input("Enter the path to your PDF document: ").strip(' "\'')
        if os.path.exists(pdf_input) and pdf_input.lower().endswith(".pdf"):
            pdf_path = pdf_input
        else:
            print(f"[Error] PDF file not found at '{pdf_input}'. Please check the path.")
            sys.exit(1)

    # Step 4: Extract and Load PDF Document
    print("\n" + "=" * 70)
    print("  STEP 4: Extract and Load PDF Document")
    print("=" * 70)

    docs = []
    if PyPDFLoader:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
        except Exception as e:
            print(f"[Warning] PyPDFLoader encountered: {e}. Attempting fallback...")

    if not docs and pypdf:
        reader = pypdf.PdfReader(pdf_path)
        for p_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            docs.append(Document(page_content=text, metadata={"source": pdf_path, "page": p_idx + 1}))

    if not docs:
        print("[Error] Failed to extract text from PDF.")
        sys.exit(1)

    total_chars = sum(len(doc.page_content) for doc in docs)
    print(f"Loaded {len(docs)} pages with a total of {total_chars} characters from '{os.path.basename(pdf_path)}'.")

    # Step 5: Text Chunking
    print("\n" + "=" * 70)
    print("  STEP 5: Text Chunking (RecursiveCharacterTextSplitter)")
    print("=" * 70)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    document_chunks = text_splitter.split_documents(docs)
    print(f"Split the document into {len(document_chunks)} discrete chunks.")

    # Step 6: Initialize Embeddings and Chroma Vector Store
    print("\n" + "=" * 70)
    print("  STEP 6: Initialize Embeddings & Chroma Vector Store")
    print("=" * 70)

    embedding_model = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=embedding_model,
        google_api_key=api_key
    )

    persist_directory = "./chroma_db"
    os.makedirs(persist_directory, exist_ok=True)

    safe_collection_name = re.sub(r'[^a-zA-Z0-9_-]', '_', Path(pdf_path).stem)[:60]
    if not safe_collection_name:
        safe_collection_name = "pdf_rag_collection"

    vectorstore = Chroma(
        collection_name=safe_collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    # Check if vector store already has existing indexed chunks
    existing_count = 0
    try:
        existing_count = vectorstore._collection.count()
    except Exception:
        pass

    if existing_count >= len(document_chunks):
        print(f"✓ Found existing indexed collection '{safe_collection_name}' with {existing_count} chunks.")
        re_index = input("Re-embed documents anyway? (y/N): ").strip().lower()
        should_embed = (re_index == 'y')
    else:
        should_embed = True

    if should_embed:
        batch_size = 5
        print(f"Embedding and inserting {len(document_chunks)} chunks in throttled batches of {batch_size}...")

        for i in range(0, len(document_chunks), batch_size):
            batch = document_chunks[i:i + batch_size]
            vectorstore.add_documents(batch)
            print(f"  - Successfully indexed chunks {i} to {min(i + batch_size, len(document_chunks))}")
            if i + batch_size < len(document_chunks):
                time.sleep(3)

        print("All chunks successfully embedded and stored in Chroma vector store!")
    else:
        print("Using existing Chroma vector store index.")

    # Step 7: Create Retriever and LLM
    print("\n" + "=" * 70)
    print("  STEP 7: Create Retriever and Gemini LLM")
    print("=" * 70)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm_model = os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite")
    llm = ChatGoogleGenerativeAI(
        model=llm_model,
        google_api_key=api_key,
        temperature=0.0
    )
    print(f"Retriever (top-3) and Gemini LLM ({llm_model}) successfully initialized!")

    # Step 8: Build the RAG Chain using LCEL
    print("\n" + "=" * 70)
    print("  STEP 8: Build LCEL RAG Chain")
    print("=" * 70)

    template = """You are a precise document assistant. Use only the provided document context below to answer the user's question accurately. If you do not know the answer or if it is not explicitly mentioned in the context, say that you cannot find it.

Context:
{context}

Question: {question}"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs_to_format):
        return "\n---\n".join(doc.page_content for doc in docs_to_format)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG Chain successfully constructed with LCEL!")

    # Step 9: Interactive Query Execution (STT -> LLM -> TTS)
    print("\n" + "=" * 70)
    print("  STEP 9: Interactive Query Loop: STT -> LLM -> TTS")
    print("=" * 70)
    print("You can ask questions via Speech (Microphone) or Type text.")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        print("-" * 60)
        print("Choose input mode:")
        print("  [1] Voice input (STT via Microphone)")
        print("  [2] Text input (Keyboard)")
        mode = input("Select option [1/2] (default: 1): ").strip()

        if mode == "2":
            user_query = input("\nEnter your document query: ").strip()
        else:
            user_query = capture_speech_to_text()

        if not user_query:
            print("[!] Empty query. Please try again.")
            continue

        if user_query.lower() in ["exit", "quit", "q"]:
            print("\nExiting RAG Voice Assistant. Goodbye!")
            break

        print(f"\nProcessing Query: {user_query}\n")

        # 1. Retrieve the matching chunks
        retrieved_docs = retriever.invoke(user_query)

        # 2. Display context chunks retrieved in TEXT format
        print("=== RETRIEVED CONTEXT SAMPLES ===")
        for idx, doc in enumerate(retrieved_docs, 1):
            content_preview = doc.page_content[:200].replace('\n', ' ').strip()
            page_num = doc.metadata.get('page', 'Unknown')
            print(f"Chunk {idx} (Page {page_num}): {content_preview}...")
        print("=================================\n")

        # 3. Generate answer through the LangChain pipeline
        print("=== GEMINI GENERATED ANSWER (TEXT) ===")
        answer = rag_chain.invoke(user_query)
        print(answer)
        print("======================================\n")

        # 4. Speak ONLY the final answer via TTS
        speak_answer_tts(answer)
        print()

if __name__ == "__main__":
    run_pipeline()
