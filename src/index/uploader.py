from typing import Iterable, List, Dict, Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient


class BatchUploader:
    def __init__(self, endpoint: str, api_key: str, index_name: str):
        self.client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))

    def upload(self, docs: Iterable[Dict[str, Any]], batch_size: int = 1000) -> int:
        buffer: List[Dict[str, Any]] = []
        uploaded = 0
        for doc in docs:
            buffer.append(doc)
            if len(buffer) >= batch_size:
                results = self.client.upload_documents(documents=buffer)
                uploaded += len(results)
                buffer.clear()
        if buffer:
            results = self.client.upload_documents(documents=buffer)
            uploaded += len(results)
        return uploaded