import json
import sys
from typing import Any
import uuid
import httpx
from pathlib import Path
from pydantic import BaseModel

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

from iointel.src.utilities.decorators import register_tool


class DocumentInfo(TypedDict):
    """Information about document stored in R2R."""

    id: str
    title: str
    summary: str
    metadata: dict


class Citation(TypedDict):
    """
    Represents a citation reference in the RAG response.

    The first time a citation appears, it includes the full payload.
    Subsequent appearances only include the citation ID and span information.
    """

    id: str
    payload: dict[str, Any]


class RAGResponse(TypedDict):
    answer: str
    citations: list[Citation] | None
    metadata: dict


class R2RClient(BaseModel):
    base_url: str
    api_key: str | None
    version: str
    timeout: float

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        version: str = "v3",
        timeout: float = 60.0,
    ):
        super().__init__(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            version=version,
            timeout=timeout,
        )

    async def _request(
        self, method: str, endpoint: str, timeout: float | None = None, **kw
    ) -> dict | bytes | None:
        headers = {
            "Accepts": "application/json",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        url = f"{self.base_url}/{self.version}/{endpoint}"
        async with httpx.AsyncClient(timeout=timeout or self.timeout) as client:
            response = await client.request(method, url, headers=headers, **kw)
            response.raise_for_status()
            if "application/json" in response.headers.get("Content-Type", "").lower():
                return response.json() if response.content else None
            else:
                return response.content

    @register_tool("r2r-create_document")
    async def create_document(
        self,
        content: str | bytes | Path,
        filename: str | None = None,
        id: str | uuid.UUID | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Create a new document.

        Args:
            content (str|bytes|Path): document content or path to upload.
                If it's bytes, it is uploaded as a octet-stream file content,
                if it's Path, it is uploaded as filesystem-existing file.
                Otherwise, it's uploaded as raw text document.
            filename (Optional[str]): document name (equal to id if not specified). Ignored if content is a str,
                and used to determine document type based on extension if content is bytes.
                If not given, is generated as document id (if given) or random uuid + .txt extension.
            id (Optional[str | UUID]): Optional ID to assign to the document
            metadata (Optional[dict]): Optional metadata to assign to the document
        Returns:
            ID of the created document
        """
        data: dict[str, str] = {}
        if id:
            data["id"] = str(id)
        if metadata:
            data["metadata"] = json.dumps(metadata)
        if isinstance(content, bytes):
            if not filename:
                filename = f"{id or uuid.uuid4().hex}.txt"
            files: httpx._types.RequestFiles = {
                "file": (filename, content, "application/octet-stream")
            }
            response_dict = await self._request(
                "POST",
                "documents",
                data=data,
                files=files,
            )
        elif isinstance(content, Path):
            with open(str(content), "rb") as fsFile:
                files: httpx._types.RequestFiles = {
                    "file": (content.name, fsFile, "application/octet-stream")
                }
                response_dict = await self._request(
                    "POST",
                    "documents",
                    data=data,
                    files=files,
                )
        else:
            data["raw_text"] = str(content) if not isinstance(content, str) else content
            response_dict = await self._request("POST", "documents", data=data)
        return response_dict["results"]["document_id"]

    @register_tool("r2r-delete_document")
    async def delete_document(self, id: str | uuid.UUID) -> bool:
        """
        Delete a specific document.

        Args:
            id (str | UUID): ID of document to delete

        Returns:
            Whether document was deleted successfully
        """
        response_dict = await self._request("DELETE", f"documents/{str(id)}")
        return response_dict["results"]["success"]

    @register_tool("r2r-list_documents")
    async def list_documents(
        self,
        ids: list[str | uuid.UUID] | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[list[DocumentInfo], int]:
        """
        List documents with pagination.

        Args:
            ids (Optional[list[str | UUID]]): Optional list of document IDs to filter by
            offset (int, optional): Specifies the number of objects to skip. Defaults to 0.
            limit (int, optional): Specifies a limit on the number of objects to return, ranging between 1 and 100. Defaults to 100.
        Returns:
            List of dictionaries describing documents found in current pagination, and the total number of documents accessible.
        """
        params: dict[str, int | list[str]] = {
            "offset": offset,
            "limit": limit,
        }
        if ids:
            params["ids"] = [str(doc_id) for doc_id in ids]

        response_dict = await self._request("GET", "documents", params=params)
        docs = [
            DocumentInfo(
                id=entry["id"],
                title=entry.get("title") or "",
                summary=entry.get("summary") or "",
                metadata=entry.get("metadata") or {},
            )
            for entry in response_dict["results"]
        ]
        return docs, response_dict["total_entries"]

    @register_tool("r2r-rag_search")
    async def rag_search(
        self,
        query: str,
        include_title_if_available: bool = False,
        include_web_search: bool = False,
        timeout: float = 300.0,
    ) -> RAGResponse:
        """
        Conducts a Retrieval Augmented Generation (RAG) search.

        Args:
            query (str): The search query.
            include_title_if_available (Optional[bool]): Include document titles in responses when available.
            include_web_search (Optional[bool]): Include web search results provided to the LLM.
            timeout (Optional[float]): Override default timeout when performing the search.

        Returns:
            Dictionary with the result of the search and citations it found if any.
        """
        payload = {
            "query": query,
            "include_title_if_available": include_title_if_available,
            "include_web_search": include_web_search,
        }
        response_dict = await self._request(
            "POST", "retrieval/rag", json=payload, timeout=max(timeout, self.timeout)
        )
        answer = response_dict["results"]
        citations: dict[str, Citation] = {}
        for entry in answer.get("citations") or ():
            if not entry.get("payload"):
                continue
            if citations.get(entry["id"], {}).get("payload"):
                continue
            citations[entry["id"]] = Citation(id=entry["id"], payload=entry["payload"])
        return RAGResponse(
            answer=answer["generated_answer"],
            citations=list(citations.values()),
            metadata=answer.get("metadata") or {},
        )
