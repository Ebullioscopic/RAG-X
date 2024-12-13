import json
from models import AdvancedProductSearchModel

def main():
    # Load product data
    with open('/Users/admin63/Python-Programs/RAG-X/data/products.json', 'r') as file:
        products_data = json.load(file)

    # Initialize search model
    search_model = AdvancedProductSearchModel(products_data)

    # User query
    query = input("Enter your search query: ")
    results = search_model.search_products(query)

    # Display results
    for product in results:
        print(json.dumps(product, indent=4))

if __name__ == "__main__":
    main()
