import asyncio
import hashlib
import json
from pathlib import Path
from typing import Optional
import aiofiles
from pydantic import BaseModel
import structlog

logger = structlog.get_logger(__name__)

class ParsedDocument(BaseModel):
    content_hash: str
    source_path: str
    markdown_text: str
    page_count: int
    tables: list[dict]
    parse_errors: list[str]

# Global cache directory for parsed documents
CACHE_DIR = Path(".cache/ingestion")

def _compute_hash_sync(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def _parse_with_docling_sync(file_path: Path) -> tuple[str, int, list[dict]]:
    """Synchronous function to wrap docling's parser."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        logger.error("docling is not installed")
        raise ImportError("docling is required for document parsing")

    converter = DocumentConverter()
    result = converter.convert(str(file_path))
    
    # Docling returns a DocumentConversionResult
    document = result.document
    
    # Export to markdown
    markdown_text = document.export_to_markdown()
    
    # Try to extract page count if available (docling structure might vary, falling back to 0)
    page_count = 0
    if hasattr(document, "pages"):
        page_count = len(document.pages)
    
    # Extract tables if available (docling exports tables in markdown or as objects)
    tables = []
    if hasattr(document, "tables"):
        for table in document.tables:
            # Very simplified table extraction as a list of dicts
            tables.append({"type": "table", "content": table.export_to_markdown()})
            
    return markdown_text, page_count, tables

async def parse_document(file_path: Path) -> ParsedDocument:
    """Parse a PDF/DOCX file into structured markdown with caching."""
    if not file_path.exists():
        return ParsedDocument(
            content_hash="",
            source_path=str(file_path),
            markdown_text="",
            page_count=0,
            tables=[],
            parse_errors=[f"File not found: {file_path}"]
        )

    # Read file and compute hash
    try:
        async with aiofiles.open(file_path, "rb") as f:
            content = await f.read()
        content_hash = await asyncio.to_thread(_compute_hash_sync, content)
    except Exception as e:
        logger.error("file_read_error", path=str(file_path), error=str(e))
        return ParsedDocument(
            content_hash="",
            source_path=str(file_path),
            markdown_text="",
            page_count=0,
            tables=[],
            parse_errors=[f"Failed to read file: {e}"]
        )

    # Check cache
    cache_file = CACHE_DIR / f"{content_hash}.json"
    if cache_file.exists():
        logger.info("ingestion_cache_hit", hash=content_hash)
        try:
            async with aiofiles.open(cache_file, "r", encoding="utf-8") as f:
                data = json.loads(await f.read())
            # Restore from cache
            return ParsedDocument(**data)
        except Exception as e:
            logger.warning("ingestion_cache_read_failed", error=str(e))
            # Fall through to re-parse if cache read fails

    # Parse with docling
    logger.info("ingestion_parsing_started", path=str(file_path))
    try:
        markdown_text, page_count, tables = await asyncio.to_thread(_parse_with_docling_sync, file_path)
        
        doc = ParsedDocument(
            content_hash=content_hash,
            source_path=str(file_path),
            markdown_text=markdown_text,
            page_count=page_count,
            tables=tables,
            parse_errors=[]
        )
        
        # Save to cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(cache_file, "w", encoding="utf-8") as f:
            await f.write(doc.model_dump_json())
            
        logger.info("ingestion_parsing_success", path=str(file_path))
        return doc
        
    except Exception as e:
        logger.error("ingestion_parsing_failed", path=str(file_path), error=str(e))
        return ParsedDocument(
            content_hash=content_hash,
            source_path=str(file_path),
            markdown_text="",
            page_count=0,
            tables=[],
            parse_errors=[f"Docling parsing failed: {str(e)}"]
        )
