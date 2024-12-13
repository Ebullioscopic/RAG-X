import re
import functools
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor
from cachetools import LRUCache, cached
from utils import get_thread_count, SPEC_PATTERNS


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
    def search_products(self, query, threshold=0.4, max_results=10):
        """
        Search for products based on the query and rank them by similarity score
        
        :param query: User's search query
        :param threshold: Matching threshold
        :param max_results: Maximum number of results to return
        :return: List of matching products with similarity scores as a JSON list
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, self.product_embeddings)[0]

        matched_products = []
        with ThreadPoolExecutor(max_workers=get_thread_count()) as executor:
            futures = [
                executor.submit(self._semantic_matching, query, similarities, threshold),
                executor.submit(self._keyword_matching, query),
                executor.submit(self._advanced_price_matching, query),
                executor.submit(self._specification_matching, query)
            ]
            for future in futures:
                matched_products.extend(future.result())

        # Deduplicate and rank results
        ranked_results = self._deduplicate_and_sort(matched_products, max_results)

        # Convert to JSON-friendly format
        return [
            {
                "product": result[0],
                "similarity_score": result[1]
            }
            for result in ranked_results
        ]

    def _semantic_matching(self, query, similarities, threshold):
        return [
            (self.products[idx], float(similarity))
            for idx, similarity in enumerate(similarities)
            if similarity > threshold
        ]

    def _keyword_matching(self, query):
        matched_products = []
        query_lower = query.lower()
        for product in self.products:
            for category, keywords in self.query_categories.items():
                if any(keyword in query_lower for keyword in keywords):
                    keyword_score = sum(1 for keyword in keywords if keyword in query_lower)
                    matched_products.append((product, 0.5 + (keyword_score * 0.1)))
        return matched_products

    def _advanced_price_matching(self, query):
        matched_products = []
        price_patterns = [
            r'(?:phones?\s*)?under\s*(?:\u20b9|INR)?\s*(\d+)',
            r'between\s*(?:\u20b9|INR)?\s*(\d+)\s*(?:to|-)\s*(?:\u20b9|INR)?(\d+)',
        ]

        for pattern in price_patterns:
            price_match = re.search(pattern, query, re.IGNORECASE)
            if price_match:
                min_price, max_price = 0, int(price_match.group(1))
                if len(price_match.groups()) > 1:
                    min_price = int(price_match.group(1))
                    max_price = int(price_match.group(2))

                for product in self.products:
                    product_price = int(product['price']['cost'].replace(',', ''))
                    if min_price <= product_price <= max_price:
                        matched_products.append((product, 0.7))
        return matched_products

    def _specification_matching(self, query):
        matched_products = []
        query_lower = query.lower()

        for product in self.products:
            description = product.get('description', '').lower()
            spec_score = sum(
                0.3 for spec_type, pattern in SPEC_PATTERNS.items()
                if spec_type in query_lower and re.search(pattern, description)
            )
            if spec_score > 0:
                matched_products.append((product, spec_score))
        return matched_products

    def _deduplicate_and_sort(self, matched_products, max_results):
        unique_matches = {}
        for product, score in matched_products:
            key = (product['name'], product['price']['cost'], product['description'])
            if key not in unique_matches or unique_matches[key][1] < score:
                unique_matches[key] = (product, score)
        return sorted(unique_matches.values(), key=lambda x: x[1], reverse=True)[:max_results]
