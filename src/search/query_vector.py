from azure.search.documents.models import VectorizedQuery

class VectorSearcher:
    """
    Search for vectors in an Azure Search index using a vector search configuration.
    """

    def __init__(self, search_client, openai_client, deployment_name ="text-embedding-small-3"):
        self.search_client = search_client 
        self.openai_client = openai_client
        self.deployment_name = deployment_name

    def generate_embedding(self, query_text):
        """"
        Generate an embedding for the given query text using OpenAI's API.
        """

        response = self.openai_client.embeddings.create(
            input=query_text,
            model=self.deployment_name,
        )
        return response.data[0].embedding

    def vector_search(self, query_text, top_k=5, vector_field=None):
        """
        Search for vectors in the index that are similar to the query text.
        """
        embedding = self.generate_embedding(query_text)
        if vector_field is None:
            vector_field = ["chunk_vector"]

        vector_query = VectorizedQuery(
            vector=embedding,
            k_nearest_neighbors=top_k,
            fields=",".join(vector_field),
        )

        # Execute search
        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=top_k,
        )

        # Return SDK results as a list; caller can format as needed
        return list(results)

    def multi_vector_search(self, query_text, top_k = 5):
        """
        Search for vectors in the index that are similar to the query text.
        """

        # generate embedding
        query_vector = self.generate_embedding(query_text)

        # define vector field 

        vector_queries = [
            VectorizedQuery(
                vector = query_vector,
                k_nearest_neighbors = top_k,
                fields = "chunk_vector",
            )
            ,
            VectorizedQuery(
                vector = query_vector,
                k_nearest_neighbors = top_k,
                fields = "chunk_vector_2",
            )
        ]

        results = self.search_client.search(
            search_text=None, 
            vector_queries = vector_queries,
            top = top_k
        )

        return list(results)

    def filtered_vector_search(self, query_text, filter_expression, top_k = 5):
        """
        Search for vectors in the index that are similar to the query text and filter the results.
        """
        
        """
        Args:
        query_text = search query 
        filter_expression: OData filter (e.g., "category eq 'Computers'")
        top_k = number of top results to return
        """

        query_vector = self.generate_embedding(query_text)

        vector_query = VectorizedQuery(
            vector = query_vector,
            k_nearest_neighbors = 50, # over fetch for extra sample 
            fields = "chunk_vector"

        )
        results = self.search_client.search(
            search_text = None, 
            vector_queries = [vector_query],
            filter = filter_expression,
            top = top_k
        )
    
        return list(results)    


    def vector_search_with_facets(self, query_text, facet_fields, top_k = 5):
        """
        Search for vectors in the index that are similar to the query text and apply facets.
        """

        query_vector = self.generate_embedding(query_text)

        vector_query = VectorizedQuery(
            vector = query_vector,
            k_nearest_neighbors = top_k, # over fetch for extra sample 
            fields = "chunk_vector"
        )

        results = self.search_client.search(
            search_text = None,
            vector_queries = [vector_query],
            facets = facet_fields,
            top = top_k 
        )

        return list(results)


"""
# Usage
        searcher = VectorSearcher(search_client, openai_client)

        # Simple vector search
        results = searcher.vector_search(
            query_text="laptop for machine learning",
            top=10
        )

        for doc in results:
            print(f"{doc['title']} (score: {doc['score']:.4f})")

        # Filtered vector search
        filtered_results = searcher.filtered_vector_search(
            query_text="gaming computer",
            filter_expression="price le 2000 and category eq 'Computers'",
            top=10
        )

        # Search with facets
        faceted_results = searcher.vector_search_with_facets(
            query_text="laptop",
            facets=["category", "price,values:500|1000|1500|2000"],
            top=10
        )

        print(f"Categories: {faceted_results['facets']['category']}")

"""