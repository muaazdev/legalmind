"""
Document Ingestion Pipeline
- PDF parsing
- Semantic chunking with overlap
- Metadata enrichment
"""
import os
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
import tiktoken


@dataclass
class DocumentMetadata:
    """Metadata for each document chunk"""
    doc_id: str
    source_file: str
    chunk_index: int
    total_chunks: int
    doc_type: str = "legal_document"
    client_id: Optional[str] = None
    date_created: str = field(default_factory=lambda: datetime.now().isoformat())
    page_number: Optional[int] = None


class DocumentIngestionPipeline:
    """
    Modular ingestion pipeline for legal documents.
    Handles PDF/text parsing, semantic chunking, and metadata enrichment.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 51,
        encoding_name: str = "cl100k_base"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # Initialize text splitter with token-based chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._token_length,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _token_length(self, text: str) -> int:
        """Calculate token length using tiktoken"""
        return len(self.encoding.encode(text))
    
    def _generate_doc_id(self, content: str, source: str) -> str:
        """Generate unique document ID based on content hash"""
        hash_input = f"{source}:{content[:500]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _extract_metadata_from_filename(self, filename: str) -> Dict[str, Any]:
        """Extract metadata hints from filename"""
        # Example: "contract_clientA_2024.pdf" -> {doc_type: "contract", client_id: "clientA"}
        parts = os.path.splitext(filename)[0].lower().split("_")
        metadata = {}
        
        doc_types = ["contract", "case", "agreement", "memo", "brief", "filing"]
        for part in parts:
            if part in doc_types:
                metadata["doc_type"] = part
            elif part.startswith("client"):
                metadata["client_id"] = part
        
        return metadata
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load document from file path"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext in [".txt", ".md"]:
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        return loader.load()
    
    def chunk_documents(
        self,
        documents: List[Document],
        source_file: str,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Split documents into chunks with enriched metadata.
        """
        # Extract base metadata from filename
        file_metadata = self._extract_metadata_from_filename(os.path.basename(source_file))
        
        # Merge with additional metadata
        if additional_metadata:
            file_metadata.update(additional_metadata)
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Enrich each chunk with metadata
        enriched_chunks = []
        doc_id = self._generate_doc_id(
            documents[0].page_content if documents else "",
            source_file
        )
        
        for idx, chunk in enumerate(chunks):
            # Create comprehensive metadata
            chunk.metadata.update({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_{idx}",
                "source_file": source_file,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "doc_type": file_metadata.get("doc_type", "legal_document"),
                "client_id": file_metadata.get("client_id"),
                "ingestion_date": datetime.now().isoformat(),
                "token_count": self._token_length(chunk.page_content)
            })
            
            enriched_chunks.append(chunk)
        
        return enriched_chunks
    
    def process_file(
        self,
        file_path: str,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Complete pipeline: load -> chunk -> enrich metadata
        """
        # Load document
        documents = self.load_document(file_path)
        
        # Chunk and enrich
        chunks = self.chunk_documents(documents, file_path, additional_metadata)
        
        return chunks
    
    def process_directory(
        self,
        directory_path: str,
        file_extensions: List[str] = [".pdf", ".txt"]
    ) -> List[Document]:
        """
        Process all documents in a directory
        """
        all_chunks = []
        
        for filename in os.listdir(directory_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in file_extensions:
                file_path = os.path.join(directory_path, filename)
                try:
                    chunks = self.process_file(file_path)
                    all_chunks.extend(chunks)
                    print(f"✓ Processed: {filename} ({len(chunks)} chunks)")
                except Exception as e:
                    print(f"✗ Error processing {filename}: {e}")
        
        return all_chunks


# Factory function for easy instantiation
def create_ingestion_pipeline(
    chunk_size: int = 512,
    chunk_overlap: int = 51
) -> DocumentIngestionPipeline:
    """Factory function to create ingestion pipeline"""
    return DocumentIngestionPipeline(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
