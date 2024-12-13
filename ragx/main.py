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
    max_results = int(input("Enter the number of results to display: "))
    
    results = search_model.search_products(query, max_results=max_results)

    # Display results in ranked order
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
