import re
import functools
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor
from cachetools import LRUCache, cached
import json
import numpy as np


class AdvancedProductSearchModel:
    def __init__(self, products_data):
        """
        Initialize the advanced product search model
        
        :param products_data: List of JSON objects containing product information
        """
        self.products = products_data
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.product_embeddings = self._compute_comprehensive_embeddings()
        self.cache = LRUCache(maxsize=1000)

    def _compute_comprehensive_embeddings(self):
        """
        Compute embeddings for all products
        
        :return: List of comprehensive embeddings for all products
        """
        product_texts = [
            " ".join(filter(bool, [
                product.get('name', ''),
                product.get('description', ''),
                product.get('company', ''),
                str(product.get('price', {})),
                " ".join([offer.get('description', '') for offer in product.get('offers', [])])
            ]))
            for product in self.products
        ]
        return self.model.encode(product_texts, convert_to_tensor=True)

    @cached(cache=functools.partial(LRUCache, maxsize=1000)())
    def search_products(self, query, similarity_threshold=0.4, search_depth=50, max_results=10):
        """
        Search for products based on the query and rank them by similarity score.
        
        :param query: User's search query
        :param similarity_threshold: Minimum similarity score for results to be included.
        :param search_depth: Depth of the search, i.e., how many results to evaluate before ranking.
        :param max_results: Maximum number of results to return.
        :return: List of matching products with similarity scores as a JSON list.
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, self.product_embeddings)[0]

        # Filter based on similarity threshold
        raw_matches = [
            (self.products[idx], float(similarity))
            for idx, similarity in enumerate(similarities)
            if similarity >= similarity_threshold
        ]

        # Apply search depth constraint
        filtered_matches = raw_matches[:search_depth]

        # Rank results by similarity scores
        sorted_matches = sorted(filtered_matches, key=lambda x: x[1], reverse=True)

        # Return top `max_results`
        return [
            {
                "product": match[0],
                "similarity_score": match[1]
            }
            for match in sorted_matches[:max_results]
        ]

    def _dual_phase_path_optimizer(self, similarities, threshold):
        """
        Dual Phase Optimization: Samples long paths for global insights 
        and refines shorter paths for detailed features.
        """
        long_path_matches = [
            (self.products[idx], float(similarity))
            for idx, similarity in enumerate(similarities)
            if similarity > threshold
        ]
        # Focus on global insights with higher threshold
        refined_matches = [
            match for match in long_path_matches if match[1] > threshold + 0.1
        ]
        return refined_matches

    def _density_adaptive_path_sampling(self, matches, similarities):
        """
        Adaptively samples paths based on density within the graph.
        Critical edges are retained to preserve graph topology.
        """
        density_scores = [
            (match[0], match[1] + (similarities[idx].sum().item() / len(similarities)))
            for idx, match in enumerate(matches)
        ]

        # Retain high-density paths
        retained_matches = [
            match for match in density_scores if match[1] > np.mean([m[1] for m in density_scores])
        ]
        return retained_matches

    def _semantic_conscious_path_structuring(self, matches):
        """
        Aligns paths with natural language patterns for semantic coherence.
        """
        semantic_grouped = {}
        for product, score in matches:
            group_key = product.get('category', 'default')
            if group_key not in semantic_grouped:
                semantic_grouped[group_key] = []
            semantic_grouped[group_key].append((product, score))

        # Select the most relevant paths from grouped categories
        final_results = []
        for group_key, group in semantic_grouped.items():
            sorted_group = sorted(group, key=lambda x: x[1], reverse=True)
            final_results.extend(sorted_group)
        return final_results

    def _get_contextual_density(self, text):
        """
        Calculates contextual density scores for token groups.
        """
        token_embeddings = self.model.encode(text.split(), convert_to_tensor=True)
        scores = util.pytorch_cos_sim(token_embeddings, token_embeddings).sum(dim=1)
        return scores.mean().item()

    def _decode_to_sentence_embeddings(self, query):
        """
        Decodes query tokens into sentence embeddings for forwarding context to LLMs.
        """
        return self.model.encode(query, convert_to_tensor=True)

    @cached(cache=functools.partial(LRUCache, maxsize=1000)())
    def search_sub_features(self, query, feature_key, similarity_threshold=0.4):
        """
        Search for specific sub-features of the products in the database.

        :param query: User's search query
        :param feature_key: Specific feature to search in the product list.
        :param similarity_threshold: Minimum similarity score for results to be included.
        :return: Filtered JSON results based on the feature key.
        """
        filtered_products = [
            {
                feature_key: product.get(feature_key, "Feature not available"),
                "product": product.get('name', 'Unnamed product')
            }
            for product in self.products if feature_key in product
        ]
        query_embedding = self._decode_to_sentence_embeddings(query)
        feature_texts = [json.dumps(prod) for prod in filtered_products]
        feature_embeddings = self.model.encode(feature_texts, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, feature_embeddings)[0]

        # Filter by similarity threshold
        results = [
            (filtered_products[idx], float(similarity))
            for idx, similarity in enumerate(similarities)
            if similarity >= similarity_threshold
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "feature": result[0],
                "similarity_score": result[1]
            }
            for result in results[:10]
        ]


# Example usage
if __name__ == "__main__":
    with open('/Users/admin63/Python-Programs/RAG-X/data/products.json', 'r') as file:
        products_data = json.load(file)

    search_model = AdvancedProductSearchModel(products_data)
    #query = "Smartphones with 64MP camera under ₹20,000"
    query = "Which is better S21 Ultra or S21+?"
    # Search with similarity threshold and depth
    results = search_model.search_products(query, similarity_threshold=0.6, search_depth=20, max_results=5)

    for result in results:
        print(json.dumps(result, indent=0))
