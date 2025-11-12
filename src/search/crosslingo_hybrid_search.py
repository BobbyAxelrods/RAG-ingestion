from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from typing import Dict, Any

class CrossLingualHybridSearcher:
    """Multilingual hybrid search with language-aware routing."""
    
    def __init__(self, search_client, embedding_function, language_detector):
        self.search_client = search_client
        self.embedding_function = embedding_function  # Multilingual model
        self.language_detector = language_detector
    
    def search(
        self,
        query: str,
        top: int = 10,
        target_languages: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute cross-lingual hybrid search.
        
        Args:
            query: Search query (any language)
            top: Number of results
            target_languages: Limit search to specific languages (e.g., ['en', 'fr'])
        """
        # Detect query language
        query_language = self.language_detector.detect(query)
        
        # Generate multilingual embedding
        # Use multilingual model (e.g., text-embedding-3-large supports 100+ languages)
        query_vector = self.embedding_function(query)
        
        # Determine BM25 strategy based on language match
        # If querying in English but documents are mixed languages:
        # - BM25: Search in query language field only
        # - Vector: Search across all languages
        
        # Create language-specific filter
        language_filter = None
        if target_languages:
            language_filter = self._create_language_filter(target_languages)
        
        # Execute hybrid search
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=50,
            fields="contentVector"  # Universal multilingual vector
        )
        
        # BM25 searches language-specific field (e.g., title_en, title_fr)
        # Vector searches universal contentVector
        results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            filter=language_filter,
            top=top
        )
        
        return list(results)
    
    def _create_language_filter(self, languages: List[str]) -> str:
        """Create filter for specific languages."""
        lang_filters = [f"language eq '{lang}'" for lang in languages]
        return ' or '.join(lang_filters)

# Usage with Azure Text Analytics for language detection
language_endpoint = "https://YOUR_TEXTANALYTICS.cognitiveservices.azure.com/"
language_key = "YOUR_KEY"

language_client = TextAnalyticsClient(
    endpoint=language_endpoint,
    credential=AzureKeyCredential(language_key)
)

def detect_language(text: str) -> str:
    """Detect language using Azure Text Analytics."""
    response = language_client.detect_language(documents=[{"id": "1", "text": text}])
    return response[0].primary_language.iso6391_name

cross_lingual_searcher = CrossLingualHybridSearcher(
    search_client,
    get_embedding,  # Must be multilingual model
    language_detector=detect_language
)

# Query in French, find English/French/Spanish results
results = cross_lingual_searcher.search(
    "ordinateur portable pour le montage vidéo",  # French query
    target_languages=['en', 'fr', 'es']
)