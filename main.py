import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
import difflib

class AdvancedProductSearchModel:
    def __init__(self, products_data):
        """
        Initialize the advanced product search model
        
        :param products_data: List of JSON objects containing product information
        """
        self.products = products_data
        # Use a more advanced sentence transformer model
        self.model = SentenceTransformer('all-mpnet-base-v2')
        
        # Precompute comprehensive embeddings
        self.product_embeddings = self._compute_comprehensive_embeddings()
        
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
        Compute more detailed embeddings for products
        
        :return: List of comprehensive embeddings for all products
        """
        product_texts = []
        for product in self.products:
            # Combine all possible information for a comprehensive embedding
            ram_match = re.search(r'(\d+)\s*GB RAM', product.get('description', ''))
            storage_match = re.search(r'(\d+)\s*GB ROM', product.get('description', ''))
            
            text_components = [
                product.get('name', ''),
                product.get('description', ''),
                product.get('company', ''),
                str(product.get('price', {})),
                # Add offer descriptions
                " ".join([str(offer.get('description', '')) for offer in product.get('offers', [])]),
                # Add technical specifications
                f"RAM: {ram_match.group(1) if ram_match else ''}",
                f"Storage: {storage_match.group(1) if storage_match else ''}"
            ]
            product_texts.append(" ".join(filter(bool, text_components)))
        
        return self.model.encode(product_texts, convert_to_tensor=True)
    
    def search_products(self, query):
        """
        Advanced search method with multiple matching strategies
        
        :param query: User's search query
        :return: List of matching products with relevance scores
        """
        # Semantic similarity search
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, self.product_embeddings)[0]
        
        # Initialize match strategies
        matched_products = []
        
        # 1. Semantic Similarity Matching
        semantic_matches = self._semantic_matching(query, similarities)
        matched_products.extend(semantic_matches)
        
        # 2. Keyword-based Matching
        keyword_matches = self._keyword_matching(query)
        matched_products.extend(keyword_matches)
        
        # 3. Price Range Matching
        price_matches = self._advanced_price_matching(query)
        matched_products.extend(price_matches)
        
        # 4. Specification-based Matching
        spec_matches = self._specification_matching(query)
        matched_products.extend(spec_matches)
        
        # Remove duplicates and sort by relevance
        def _product_key(product_info):
            """
            Create a hashable key for a product to identify unique products
            """
            product, _ = product_info
            # Use a combination of unique identifiers
            return (
                product.get('name', ''),
                product.get('price', {}).get('cost', ''),
                product.get('description', '')
            )
        
        # Remove duplicates while preserving the highest score for each unique product
        unique_matches = {}
        for product_info in matched_products:
            key = _product_key(product_info)
            current_score = product_info[1]
            
            if key not in unique_matches or current_score > unique_matches[key][1]:
                unique_matches[key] = product_info
        
        # Sort by relevance and return products
        sorted_matches = sorted(unique_matches.values(), key=lambda x: x[1], reverse=True)
        
        return [product for product, score in sorted_matches]
    
    def _semantic_matching(self, query, similarities, threshold=0.4):
        """
        Perform semantic matching with improved threshold strategy
        
        :param query: User's search query
        :param similarities: Precomputed similarity scores
        :param threshold: Matching threshold
        :return: List of semantically matched products
        """
        matched_products = []
        for idx, similarity in enumerate(similarities):
            # Adaptive thresholding based on query length
            adaptive_threshold = threshold - (0.1 if len(query) > 30 else 0)
            if similarity > adaptive_threshold:
                matched_products.append((self.products[idx], float(similarity)))
        return matched_products
    
    def _keyword_matching(self, query):
        """
        Match products based on specific keywords
        
        :param query: User's search query
        :return: List of keyword-matched products
        """
        matched_products = []
        
        # Extract query keywords
        query_lower = query.lower()
        
        for product in self.products:
            # Check for specific query categories
            for category, keywords in self.query_categories.items():
                if any(keyword in query_lower for keyword in keywords):
                    # Higher score for multiple keyword matches
                    keyword_score = sum(1 for keyword in keywords if keyword in query_lower)
                    matched_products.append((product, 0.5 + (keyword_score * 0.1)))
                    break
        
        return matched_products
    
    def _advanced_price_matching(self, query):
        """
        Advanced price matching with multiple strategies
        
        :param query: User's search query
        :return: List of price-matched products
        """
        # Advanced price extraction
        price_patterns = [
            r'(?:phones?\s*)?under\s*(?:₹|INR)?\s*(\d+)',
            r'between\s*(?:₹|INR)?\s*(\d+)\s*(?:to|-)\s*(?:₹|INR)?(\d+)',
            r'(?:₹|INR)?\s*(\d+)\s*(?:to|-)\s*(?:₹|INR)?(\d+)'
        ]
        
        matched_products = []
        
        for pattern in price_patterns:
            price_match = re.search(pattern, query, re.IGNORECASE)
            if price_match:
                # Extract price range
                if len(price_match.groups()) == 1:
                    max_price = int(price_match.group(1))
                    min_price = 0
                else:
                    min_price = int(price_match.group(1))
                    max_price = int(price_match.group(2))
                
                # Filter products by price
                for product in self.products:
                    try:
                        # Safely extract and parse price
                        price_str = str(product['price'].get('cost', 'Price not available')).replace(',', '')
                        
                        # Only convert to int if it looks like a valid price
                        if price_str.isdigit():
                            product_price = int(price_str)
                            
                            if min_price <= product_price <= max_price:
                                matched_products.append((product, 0.7))
                    except (ValueError, TypeError, KeyError):
                        # Skip products with invalid price format
                        continue
        
        return matched_products
    
    def _specification_matching(self, query):
        """
        Match products based on specific technical specifications
        
        :param query: User's search query
        :return: List of specification-matched products
        """
        matched_products = []
        query_lower = query.lower()
        
        # Specification extraction patterns
        spec_patterns = {
            'ram': r'(\d+)\s*gb\s*ram',
            'storage': r'(\d+)\s*gb\s*(?:rom|storage)',
            'camera': r'(\d+)\s*mp\s*camera',
            'battery': r'(\d+)\s*mah\s*battery'
        }
        
        for product in self.products:
            spec_score = 0
            description = product.get('description', '').lower()
            
            # Check each specification type
            for spec_type, pattern in spec_patterns.items():
                # Check if specification is in the query
                if spec_type in query_lower:
                    # Find specification in description
                    spec_match = re.search(pattern, description)
                    if spec_match:
                        spec_value = spec_match.group(1)
                        # Check if query contains the specific value
                        if spec_value in query_lower:
                            spec_score += 0.3
            
            if spec_score > 0:
                matched_products.append((product, spec_score))
        
        return matched_products
    
    def extract_detailed_information(self, product, query):
        """
        Extract and highlight specific information based on the query
        
        :param product: Product dictionary
        :param query: User's search query
        :return: Dictionary of extracted information
        """
        extracted_info = {
            'name': product.get('name', ''),
            'price': product.get('price', {}),
            'platform': product.get('platform', 'Unknown'),
            'description': product.get('description', '')
        }
        
        # Specific information extraction based on query
        query_lower = query.lower()
        
        # Offers and discounts
        if any(keyword in query_lower for keyword in ['offer', 'discount', 'cashback', 'exchange']):
            extracted_info['offers'] = self._extract_best_offers(product)
        
        # Specification details
        if any(keyword in query_lower for keyword in ['spec', 'specification', 'detail']):
            extracted_info['technical_specs'] = self._extract_technical_specs(product)
        
        # Variants
        if any(keyword in query_lower for keyword in ['variant', 'version', 'configuration']):
            extracted_info['variants'] = self._find_similar_variants(product)
        
        return extracted_info
    
    def _extract_best_offers(self, product):
        """
        Extract the best offers from a product
        
        :param product: Product dictionary
        :return: List of best offers
        """
        offers = product.get('offers', [])
        
        def parse_offer_amount(offer):
            """
            Parse offer amount, handling comma-separated strings
            
            :param offer: Offer dictionary
            :return: Numeric value of offer amount
            """
            amount = offer.get('offer-amount', '0')
            
            # Handle 'Unknown' or non-numeric amounts
            if amount == 'Unknown':
                return 0
            
            # Remove commas and convert to float
            try:
                return float(str(amount).replace(',', ''))
            except (ValueError, TypeError):
                return 0
        
        # Sort offers by parsed offer amount
        sorted_offers = sorted(
            offers, 
            key=parse_offer_amount, 
            reverse=True
        )
        
        # Return top 3 offers
        return sorted_offers[:3]
    
    def _extract_technical_specs(self, product):
        """
        Extract detailed technical specifications
        
        :param product: Product dictionary
        :return: Dictionary of technical specifications
        """
        description = product.get('description', '')
        
        def safe_group_extract(match):
            """
            Safely extract group from regex match
            
            :param match: Regex match object or None
            :return: Matched string or 'Not specified'
            """
            if match:
                # For single group matches
                if hasattr(match, 'group'):
                    return match.group()
                # For nested matches (like camera)
                elif isinstance(match, dict):
                    return {k: v.group() if v else 'Not specified' for k, v in match.items()}
            return 'Not specified'
        
        specs = {
            'ram': re.search(r'(\d+)\s*GB RAM', description),
            'storage': re.search(r'(\d+)\s*GB ROM', description),
            'display': re.search(r'(\d+\.?\d*)\s*cm\s*\((\d+\.?\d*)\s*inch\)', description),
            'camera': {
                'main': re.search(r'(\d+)MP\s*\+\s*(\d+)MP', description),
                'front': re.search(r'(\d+)MP\s*Front\s*Camera', description)
            },
            'battery': re.search(r'(\d+)\s*mAh\s*Lithium\s*ion', description),
            'processor': re.search(r'(\w+\s*\w+\s*\d+\+?)', description)
        }
        
        # Use safe extraction method
        return {k: safe_group_extract(v) for k, v in specs.items()}
    
    def _find_similar_variants(self, product):
        """
        Find similar variants of the product
        
        :param product: Product dictionary
        :return: List of similar variants
        """
        name = product.get('name', '')
        
        # Find variants with similar names
        similar_variants = [
            p for p in self.products 
            if difflib.SequenceMatcher(None, name, p.get('name', '')).ratio() > 0.6
        ]
        
        return similar_variants

# Example usage function
def main():
    # Load product data
    with open('/Users/admin63/Python-Programs/RAG-X/data/products.json', 'r') as f:
        products_data = json.load(f)
    
    # Initialize the search model
    search_model = AdvancedProductSearchModel(products_data)
    
    # Example queries demonstrating different types of searches
    queries = [
        "What are the specifications of the SAMSUNG Galaxy F15 5G?",
        "Phones under ₹15,000",
        "What's the difference between 4 GB and 6 GB RAM versions?",
        "Best exchange offers for this phone",
        "Does this phone have a 50 MP camera?",
        "Compare variants of Samsung Galaxy F15 5G"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        
        # Search for matching products
        results = search_model.search_products(query)
        
        # Extract and display detailed information
        for product in results[:3]:  # Show top 3 results
            print("\nProduct Details:")
            detailed_info = search_model.extract_detailed_information(product, query)
            
            # Pretty print the detailed information
            for key, value in detailed_info.items():
                print(f"{key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    main()