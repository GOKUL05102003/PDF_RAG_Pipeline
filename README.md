# Document RAG Pipeline System with Voice Interaction (STT -> LLM -> TTS)

An end-to-end Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Google Gemini**, and **ChromaDB**, featuring bidirectional voice interaction:
- **Speech-to-Text (STT)**: Ask questions using your voice via microphone or audio files.
- **RAG Retrieval & Generation (LLM)**: Retrieves the most relevant document chunks and generates grounded answers using Google Gemini.
- **Dual Format Output**:
  1. **Text**: Displays the retrieved chunks (with page numbers and text snippets) and the full generated answer.
  2. **Text-to-Speech (TTS)**: Speaks **only the final answer** out loud cleanly.

---

## Architecture Flow

```
                      +-------------------+
                      |   PDF Document    |
                      +---------+---------+
                                |
                   (PyPDFLoader / pypdf)
                                |
                                v
                    +-----------------------+
                    | Chunking (Recursive)  |
                    +-----------+-----------+
                                |
             (GoogleGenerativeAIEmbeddings + Throttling)
                                |
                                v
                    +-----------------------+
                    |   Chroma VectorDB     |
                    +-----------+-----------+
                                |
      User Voice                | Context
      +--------+                | Chunks
      | (Mic)  |                |
      +---+----+                v
          | (STT)      +-----------------+
          +----------->| LCEL RAG Chain  |
     Voice Query       | (Gemini Flash)  |
                       +--------+--------+
                                |
                 +--------------+--------------+
                 |                             |
                 v                             v
      +----------------------+      +----------------------+
      |     Text Output      |      |      TTS Output      |
      | - Context Chunks     |      | - Vocalizes ONLY     |
      | - Grounded Answer    |      |   the final answer   |
      +----------------------+      +----------------------+
```

---

## Project Structure

```
d:\Study\AI\Rag\
├── .env                  # Environment variables (API keys, model configs)
├── .env.example          # Template for environment variables
├── requirements.txt      # Project dependencies
├── README.md             # Documentation and usage guide
├── rag_pipeline.py       # Standalone script matching the original notebook flow + STT/TTS
├── main.py               # Modular CLI entry point
├── src/                  # Modular package
│   ├── __init__.py
│   ├── config.py         # Configuration settings & defaults
│   ├── audio/            # Voice modules
│   │   ├── __init__.py
│   │   ├── stt.py        # Speech-to-Text (Microphone capture & audio file transcription)
│   │   └── tts.py        # Text-to-Speech (Speech synthesis for final answer)
│   └── rag/              # RAG pipeline modules
│       ├── __init__.py
│       ├── loader.py     # PDF loading (PyPDFLoader & fallback)
│       ├── splitter.py   # Text chunking (RecursiveCharacterTextSplitter)
│       ├── vectorstore.py# Embeddings & Chroma vector store creation
│       └── chain.py      # LCEL RAG chain construction & execution
├── data/                 # Directory to place your PDF documents
└── chroma_db/            # Persistent local Chroma vector database
```

---

## Models Configured

This project is configured with the Google Gemini models from the reference pipeline:
- **Embedding Model**: `gemini-embedding-001`
- **LLM Model**: `gemini-3.5-flash-lite`

You can customize these in `.env`:
```ini
EMBEDDING_MODEL=gemini-embedding-001
LLM_MODEL=gemini-3.5-flash-lite
```

---

## Setup & Configuration

### 1. Configure Your Gemini API Key
You only need **ONE** single API key. Edit `.env` in the project root:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
EMBEDDING_MODEL=gemini-embedding-001
LLM_MODEL=gemini-3.5-flash-lite
```
> **Note on API Keys:** Having two separate keys is **not mandatory**. The system automatically binds `GEMINI_API_KEY` to both LangChain and Google GenAI libraries under the hood.

### 2. Target Document
The project is configured by default with the uploaded report:
- `data/iPhone_16_Pro_and_iPhone_16_Pro_Max_PER_Sept2024.pdf` (12 pages)

---

## Speech & Audio Setup (STT & TTS)

### Native Windows STT (Zero PyAudio Dependency)
- Standard `SpeechRecognition` requires `PyAudio`, which often fails to compile on Windows (requiring Microsoft C++ Build Tools).
- This project integrates a **native `sounddevice` audio capture engine** that records directly from your microphone array without needing `PyAudio` or C++ build tools.
- It includes automatic silence detection to conclude the recording when you stop speaking.
- **Graceful Fallback**: If no microphone is connected, it lets you type your question.

### Text-to-Speech (TTS)
- Uses offline system TTS (`pyttsx3`) with SAPI5 on Windows.
- Automatically strips markdown formatting (asterisks, hashtags, bullet points, links) so speech sounds natural.
- **Speaks ONLY the final answer**, while displaying full context chunks on the console.

---

## How to Run

### Option A: Standalone Flow Script (`rag_pipeline.py`)
This script reproduces the exact 9-step pipeline from `RAG_LangChain.ipynb` from scratch, enriched with STT and TTS:
```bash
python rag_pipeline.py
```
**Steps executed:**
1. Loads environment and configures Gemini API key from `.env`.
2. Detects or prompts for a PDF document.
3. Loads document pages (`PyPDFLoader`).
4. Splits text into discrete chunks (`RecursiveCharacterTextSplitter`).
5. Embeds chunks in throttled batches to respect Gemini free-tier rate limits (`GoogleGenerativeAIEmbeddings` + `Chroma`).
6. Configures top-3 retriever and `ChatGoogleGenerativeAI` (`gemini-2.5-flash`).
7. Constructs the LCEL chain (`prompt | retriever | llm | StrOutputParser`).
8. Launches interactive STT -> LLM -> TTS loop.

### Option B: Modular CLI (`main.py`)
The modular interface supports flexible options:

1. **Interactive Mode (Voice or Text choice)**:
   ```bash
   python main.py
   ```

2. **Specify PDF directly**:
   ```bash
   python main.py --pdf "data/your_document.pdf"
   ```

3. **Start directly in Voice STT Mode**:
   ```bash
   python main.py --pdf "data/your_document.pdf" --voice
   ```

4. **Single Query with Text & Voice Output**:
   ```bash
   python main.py --pdf "data/your_document.pdf" --query "What is the main finding?"
   ```

5. **Transcribe an Audio File Query**:
   ```bash
   python main.py --pdf "data/your_document.pdf" --audio-file "question.wav"
   ```

---

## Features & Voice Details

### Speech-to-Text (STT)
- Uses `speech_recognition` with automatic ambient noise calibration.
- Supports microphone recording with dynamic phrase detection.
- Includes a graceful text-typing fallback if microphone is unavailable or silent.

### Text-to-Speech (TTS)
- Uses offline system TTS (`pyttsx3`) that requires no additional cloud API calls or latency.
- Automatically strips markdown formatting (asterisks, hashtags, links, code tags) so that the synthesized voice sounds fluent and natural.
- **Strictly speaks only the final answer**, while printing full context chunks and page numbers on the console.

