from pathlib import Path
from typing import List
from langchain_core.documents import Document

def load_pdf(pdf_path: str) -> List[Document]:
    """
    Extracts and loads document pages and metadata from a PDF file
    matching Step 4 of the reference notebook.
    """
    file_path = Path(pdf_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file does not exist at: {pdf_path}")

    # Primary loader: PyPDFLoader from langchain_community
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
    except Exception as primary_err:
        # Fallback to direct pypdf extraction if needed
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            docs = []
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                docs.append(Document(
                    page_content=text,
                    metadata={"source": str(file_path), "page": page_idx + 1}
                ))
        except Exception as fallback_err:
            raise RuntimeError(
                f"Failed to load PDF using PyPDFLoader ({primary_err}) "
                f"and fallback pypdf ({fallback_err})"
            )

    total_chars = sum(len(doc.page_content) for doc in docs)
    print(f"[Loader] Loaded {len(docs)} pages with a total of {total_chars} characters from '{file_path.name}'.")
    return docs

