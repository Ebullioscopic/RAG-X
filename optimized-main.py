import re
import functools
import psutil
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, util
from cachetools import LRUCache, cached
import json
import math


class AdvancedProductSearchModel:
    def __init__(self, products_data):
        """
        Initialize the product search model with query categorization support and multithreading.

        :param products_data: List of JSON objects containing product information
        """
        self.products = products_data
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.product_embeddings = self._compute_comprehensive_embeddings()
        self.cache = LRUCache(maxsize=1000)
        self.query_categories = {
            "product_information": [
                "specifications", "configurations", "features", "details", "availability", "colors", "variants"
            ],
            "pricing": ["price", "cost", "cheaper", "under", "affordable"],
            "offers_discounts": ["offer", "discount", "cashback", "exchange", "deal", "EMI"],
            "comparisons": ["compare", "better", "difference"],
            "warranty_services": ["warranty", "repair", "service center"],
            "delivery": ["delivery", "shipping", "pickup", "logistics"],
            "technical_queries": ["performance", "gaming", "storage", "expandability"],
            "user_reviews": ["reviews", "issues", "feedback"],
            "accessories": ["charger", "compatible accessories"],
        }

    def _get_optimal_thread_count(self):
        """
        Dynamically calculate the optimal number of threads based on system resources.

        :return: Number of threads to use
        """
        cpu_count = psutil.cpu_count(logical=True)
        memory_info = psutil.virtual_memory()
        available_memory_gb = memory_info.available / (1024 ** 3)  # Convert to GB

        # Threads are proportional to CPU cores, adjusted for memory constraints
        max_threads_by_cpu = cpu_count * 2  # Assuming hyperthreading
        max_threads_by_memory = math.floor(available_memory_gb / 0.5)  # Assuming each thread uses 0.5GB

        return min(max_threads_by_cpu, max_threads_by_memory)

    def _compute_comprehensive_embeddings(self):
        """
        Compute embeddings for all products using multithreading.

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

        thread_count = self._get_optimal_thread_count()
        print(f"Using {thread_count} threads for embeddings computation.")

        # Multithreaded embedding computation with keyword arguments
        def compute_embedding(text):
            return self.model.encode(text, convert_to_tensor=True)

        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            embeddings = list(executor.map(compute_embedding, product_texts))
        return embeddings


    def _classify_query(self, query):
        """
        Classify a query into predefined categories based on keywords and embeddings.

        :param query: User's query as a string.
        :return: Detected query category.
        """
        for category, keywords in self.query_categories.items():
            if any(keyword in query.lower() for keyword in keywords):
                return category
        return "general"

    @cached(cache=functools.partial(LRUCache, maxsize=1000)())
    def search_products(self, query, similarity_threshold=0.4, search_depth=50, max_results=10):
        """
        General search for products based on the query, with multithreading for similarity calculation.

        :param query: User's search query
        :param similarity_threshold: Minimum similarity score for results.
        :param search_depth: Depth of the search before ranking.
        :param max_results: Maximum number of results to return.
        :return: List of matching products with similarity scores as JSON.
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        thread_count = self._get_optimal_thread_count()
        print(f"Using {thread_count} threads for similarity computation.")

        def compute_similarity(index):
            return (index, float(util.cos_sim(query_embedding, self.product_embeddings[index])[0][0]))

        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            similarities = list(executor.map(compute_similarity, range(len(self.products))))

        # Filter and rank based on similarity scores
        filtered_matches = [
            (self.products[idx], score) for idx, score in similarities if score >= similarity_threshold
        ][:search_depth]

        sorted_matches = sorted(filtered_matches, key=lambda x: x[1], reverse=True)

        # Return top `max_results`
        return [
            {
                "product": match[0],
                "similarity_score": match[1]
            }
            for match in sorted_matches[:max_results]
        ]

    def handle_product_information(self, query):
        """
        Handle queries related to product information.

        :param query: User's query as a string.
        :return: Matched product information.
        """
        results = self.search_products(query, similarity_threshold=0.5, max_results=5)
        return [{"product_info": result["product"], "similarity_score": result["similarity_score"]} for result in results]

    def handle_pricing_query(self, query):
        """
        Handle queries related to product pricing.

        :param query: User's query as a string.
        :return: Price information for relevant products.
        """
        results = self.search_products(query, similarity_threshold=0.6, max_results=3)
        price_info = []
        for result in results:
            product = result["product"]
            price_info.append({
                "name": product.get("name"),
                "price": product.get("price"),
                "similarity_score": result["similarity_score"]
            })
        return price_info

    def handle_offers_discounts(self, query):
        """
        Handle queries about offers and discounts.

        :param query: User's query as a string.
        :return: List of available offers.
        """
        offers = []
        for product in self.products:
            for offer in product.get('offers', []):
                if query.lower() in offer.get('description', '').lower():
                    offers.append({
                        "product": product.get("name"),
                        "offer_details": offer.get("description"),
                        "price": product.get("price", "Price not available")
                    })
        return offers

    def handle_query(self, query):
        """
        Dynamically handle the user's query by determining its category.

        :param query: User's query as a string.
        :return: Result based on the query category.
        """
        query_category = self._classify_query(query)

        if query_category == "product_information":
            return self.handle_product_information(query)
        elif query_category == "pricing":
            return self.handle_pricing_query(query)
        elif query_category == "offers_discounts":
            return self.handle_offers_discounts(query)
        else:
            return self.search_products(query, similarity_threshold=0.4)

    def pretty_print_results(self, results):
        """
        Pretty print the results for user-friendly display.

        :param results: Results to print.
        """
        for result in results:
            print(json.dumps(result, indent=4))


# Example Usage
if __name__ == "__main__":
    with open('/Users/admin63/Python-Programs/RAG-X/data/products.json', 'r') as file:
        products_data = json.load(file)

    search_model = AdvancedProductSearchModel(products_data)

    queries = [
        "What are the specifications of the SAMSUNG Galaxy F15 5G?",
        "What is the price of the SAMSUNG Galaxy F15 5G with 6 GB RAM?",
        "Are there any exchange offers for this phone?",
        "Is there any phone under ₹15,000 with similar specifications?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = search_model.handle_query(query)
        search_model.pretty_print_results(results)
