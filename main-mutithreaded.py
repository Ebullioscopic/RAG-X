import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
import difflib
import threading
from concurrent.futures import ThreadPoolExecutor
import os
import psutil
import functools
from cachetools import LRUCache, cached

class AdvancedProductSearchModel:
    def __init__(self, products_data):
        """
        Initialize the advanced product search model
        
        :param products_data: List of JSON objects containing product information
        """
        self.products = products_data
        self.model = SentenceTransformer('all-mpnet-base-v2')

        # Precompute comprehensive embeddings
        self.product_embeddings = self._compute_comprehensive_embeddings()

        # Cache for queries to improve performance
        self.cache = LRUCache(maxsize=1000)

        # Predefined query categories and their keywords
        self.query_categories = {
            'specifications': ['specification', 'details', 'feature', 'spec'],
            'variants': ['variant', 'different', 'configuration', 'version'],
            'price': ['price', 'cost', 'cheap', 'expensive'],
            'offers': ['offer', 'discount', 'deal', 'cashback', 'exchange'],
            'platform': ['platform', 'seller', 'vendor', 'store'],
            'performance': ['performance', 'processor', 'gaming', 'speed'],
            'camera': ['camera', 'photo', 'photography'],
            'battery': ['battery', 'power', 'charging'],
            'storage': ['storage', 'memory', 'sd card', 'expandable']
        }

    def _compute_comprehensive_embeddings(self):
        """
        Compute embeddings for all products
        
        :return: List of comprehensive embeddings for all products
        """
        product_texts = []
        for product in self.products:
            ram_match = re.search(r'(\d+)\s*GB RAM', product.get('description', ''))
            storage_match = re.search(r'(\d+)\s*GB ROM', product.get('description', ''))

            text_components = [
                product.get('name', ''),
                product.get('description', ''),
                product.get('company', ''),
                str(product.get('price', {})),
                " ".join([str(offer.get('description', '')) for offer in product.get('offers', [])]),
                f"RAM: {ram_match.group(1) if ram_match else ''}",
                f"Storage: {storage_match.group(1) if storage_match else ''}"
            ]
            product_texts.append(" ".join(filter(bool, text_components)))

        return self.model.encode(product_texts, convert_to_tensor=True)

    @cached(cache=functools.partial(LRUCache, maxsize=1000)())
    def search_products(self, query, threshold=0.4, max_results=10):
        """
        Advanced search with multiple matching strategies

        :param query: User's search query
        :param threshold: Matching threshold
        :param max_results: Maximum number of results to return
        :return: List of matching products with relevance scores
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, self.product_embeddings)[0]

        matched_products = []

        # Multithreading for matching strategies
        with ThreadPoolExecutor(max_workers=self._get_thread_count()) as executor:
            futures = [
                executor.submit(self._semantic_matching, query, similarities, threshold),
                executor.submit(self._keyword_matching, query),
                executor.submit(self._advanced_price_matching, query),
                executor.submit(self._specification_matching, query)
            ]

            for future in futures:
                matched_products.extend(future.result())

        # Remove duplicates and sort by relevance
        def _product_key(product_info):
            product, _ = product_info
            return (product.get('name', ''), product.get('price', {}).get('cost', ''), product.get('description', ''))

        unique_matches = {}
        for product_info in matched_products:
            key = _product_key(product_info)
            current_score = product_info[1]
            if key not in unique_matches or current_score > unique_matches[key][1]:
                unique_matches[key] = product_info

        sorted_matches = sorted(unique_matches.values(), key=lambda x: x[1], reverse=True)
        return [product for product, score in sorted_matches[:max_results]]

    def _semantic_matching(self, query, similarities, threshold):
        matched_products = []
        for idx, similarity in enumerate(similarities):
            adaptive_threshold = threshold - (0.1 if len(query) > 30 else 0)
            if similarity > adaptive_threshold:
                matched_products.append((self.products[idx], float(similarity)))
        return matched_products

    def _keyword_matching(self, query):
        matched_products = []
        query_lower = query.lower()
        for product in self.products:
            for category, keywords in self.query_categories.items():
                if any(keyword in query_lower for keyword in keywords):
                    keyword_score = sum(1 for keyword in keywords if keyword in query_lower)
                    matched_products.append((product, 0.5 + (keyword_score * 0.1)))
                    break
        return matched_products

    def _advanced_price_matching(self, query):
        price_patterns = [
            r'(?:phones?\s*)?under\s*(?:\u20b9|INR)?\s*(\d+)',
            r'between\s*(?:\u20b9|INR)?\s*(\d+)\s*(?:to|-)\s*(?:\u20b9|INR)?(\d+)',
            r'(?:\u20b9|INR)?\s*(\d+)\s*(?:to|-)\s*(?:\u20b9|INR)?(\d+)'
        ]

        matched_products = []

        for pattern in price_patterns:
            price_match = re.search(pattern, query, re.IGNORECASE)
            if price_match:
                if len(price_match.groups()) == 1:
                    max_price = int(price_match.group(1))
                    min_price = 0
                else:
                    min_price = int(price_match.group(1))
                    max_price = int(price_match.group(2))

                for product in self.products:
                    try:
                        price_str = str(product['price'].get('cost', 'Price not available')).replace(',', '')
                        if price_str.isdigit():
                            product_price = int(price_str)
                            if min_price <= product_price <= max_price:
                                matched_products.append((product, 0.7))
                    except (ValueError, TypeError, KeyError):
                        continue
        return matched_products

    def _specification_matching(self, query):
        matched_products = []
        query_lower = query.lower()

        spec_patterns = {
            'ram': r'(\d+)\s*gb\s*ram',
            'storage': r'(\d+)\s*gb\s*(?:rom|storage)',
            'camera': r'(\d+)\s*mp\s*camera',
            'battery': r'(\d+)\s*mah\s*battery'
        }

        for product in self.products:
            spec_score = 0
            description = product.get('description', '').lower()
            for spec_type, pattern in spec_patterns.items():
                if spec_type in query_lower:
                    spec_match = re.search(pattern, description)
                    if spec_match:
                        spec_value = spec_match.group(1)
                        if spec_value in query_lower:
                            spec_score += 0.3
            if spec_score > 0:
                matched_products.append((product, spec_score))
        return matched_products

    def _get_thread_count(self):
        available_memory = psutil.virtual_memory().available / (1024 ** 2)
        cpu_count = os.cpu_count() or 4

        if available_memory < 2000:
            return max(2, cpu_count // 2)
        return cpu_count

# Example usage
if __name__ == "__main__":
    with open('/Users/admin63/Python-Programs/RAG-X/data/products.json', 'r') as f:
        products_data = json.load(f)

    search_model = AdvancedProductSearchModel(products_data)
    query = "Which is better S21 Ultra or S21+?"
    results = search_model.search_products(query, max_results=5)

    for product in results:
        print(json.dumps(product, indent=4))
