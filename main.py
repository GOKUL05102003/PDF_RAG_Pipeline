"""
Document RAG Assistant CLI with STT -> LLM -> TTS
=================================================
Orchestrates the modular RAG pipeline with Speech-to-Text and Text-to-Speech.
"""

import os
import sys
import argparse
from pathlib import Path

from src import config
from src.audio import get_user_query, speak_text, transcribe_from_file
from src.rag import (
    load_pdf,
    split_documents,
    build_vectorstore,
    create_rag_chain,
    query_rag_pipeline,
)

def find_default_pdf() -> str:
    """Finds default document or any PDF in data/ or root directory."""
    if hasattr(config, "DEFAULT_PDF_PATH") and config.DEFAULT_PDF_PATH.exists():
        return str(config.DEFAULT_PDF_PATH)
    candidates = list(config.DATA_DIR.glob("*.pdf")) + list(Path(".").glob("*.pdf"))
    if candidates:
        return str(candidates[0])
    return ""

def main():
    parser = argparse.ArgumentParser(
        description="PDF RAG Pipeline with Voice Input (STT) and Voice Output (TTS)"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to the PDF file to index and query."
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Start directly in Speech-to-Text (STT) voice input mode."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to run directly without interactive prompt."
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Transcribe an input audio file (WAV/FLAC) as the user query."
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable Text-to-Speech audio playback for the final answer."
    )
    parser.add_argument(
        "--save-audio",
        type=str,
        default=None,
        help="Optional path to save TTS answer audio (e.g. answer.wav)."
    )

    args = parser.parse_args()

    print("=" * 65)
    print("  Document RAG Assistant (LangChain + Gemini + STT/TTS)")
    print("=" * 65)

    # 1. Verify API Key
    if not config.GEMINI_API_KEY or config.GEMINI_API_KEY == "your_gemini_api_key_here":
        print("\n[!] Google Gemini API Key is missing.")
        print("Please edit the .env file and set GOOGLE_API_KEY=your_key")
        user_key = input("Or enter your API key now: ").strip()
        if not user_key:
            print("Cannot continue without API key. Exiting.")
            sys.exit(1)
        os.environ["GOOGLE_API_KEY"] = user_key
        config.GEMINI_API_KEY = user_key

    # 2. Locate PDF File
    pdf_path = args.pdf
    if not pdf_path:
        pdf_path = find_default_pdf()

    if not pdf_path or not Path(pdf_path).exists():
        print("\nNo default PDF found in workspace or data/ directory.")
        pdf_path = input("Enter path to PDF document: ").strip(' "\'')
        if not Path(pdf_path).exists():
            print(f"Error: File '{pdf_path}' does not exist.")
            sys.exit(1)

    print(f"\n[Target Document]: {pdf_path}")

    # 3. Load & Chunk Document
    docs = load_pdf(pdf_path)
    chunks = split_documents(docs)

    # 4. Build Vector Store & RAG Chain
    vectorstore = build_vectorstore(chunks)
    rag_chain, retriever = create_rag_chain(vectorstore)

    # 5. Handle Single Query Mode
    if args.query or args.audio_file:
        query_text = args.query
        if args.audio_file:
            query_text = transcribe_from_file(args.audio_file)

        if not query_text:
            print("[Error] No query text available.")
            sys.exit(1)

        result = query_rag_pipeline(rag_chain, retriever, query_text)

        print("\n=== RETRIEVED CONTEXT SAMPLES ===")
        for idx, doc in enumerate(result["retrieved_docs"], 1):
            preview = doc.page_content[:200].replace('\n', ' ').strip()
            page = doc.metadata.get("page", "Unknown")
            print(f"Chunk {idx} (Page {page}): {preview}...")
        print("=================================\n")

        print("=== GEMINI GENERATED ANSWER ===")
        print(result["answer"])
        print("===============================\n")

        if not args.no_tts:
            speak_text(result["answer"], save_to_file=args.save_audio)
        return

    # 6. Interactive Mode
    print("\nStarting Interactive Session. You can query via STT (Voice) or Text.")
    print("Commands: type 'exit', 'quit', or 'q' to end.")

    prefer_voice = args.voice

    while True:
        try:
            print("\n" + "-" * 50)
            print("Query Input Options:")
            print("  [1] Voice input (STT - Microphone)")
            print("  [2] Text input (Keyboard)")
            choice = input("Select mode [1/2] (default: 1): ").strip()
            voice_mode = (choice != "2")

            query = get_user_query(prefer_voice=voice_mode)

            if not query:
                print("Empty query. Try again.")
                continue

            if query.lower() in ["exit", "quit", "q"]:
                print("\nExiting session. Have a great day!")
                break

            result = query_rag_pipeline(rag_chain, retriever, query)

            print("\n=== RETRIEVED CONTEXT SAMPLES ===")
            for idx, doc in enumerate(result["retrieved_docs"], 1):
                preview = doc.page_content[:200].replace('\n', ' ').strip()
                page = doc.metadata.get("page", "Unknown")
                print(f"Chunk {idx} (Page {page}): {preview}...")
            print("=================================\n")

            print("=== GEMINI GENERATED ANSWER (TEXT) ===")
            print(result["answer"])
            print("======================================\n")

            if not args.no_tts:
                speak_text(result["answer"], save_to_file=args.save_audio)

        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break
        except Exception as err:
            print(f"\n[Error during query processing]: {err}")

if __name__ == "__main__":
    main()

