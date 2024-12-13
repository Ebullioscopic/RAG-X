# Advanced Product Search System

This repository implements an advanced product search system using a **SentenceTransformer** model for semantic similarity and multiple search strategies (e.g., semantic, keyword, price, specification matching). The system ranks and returns search results as JSON with similarity scores.

## Features
- Multi-strategy product search:
  - Semantic matching using sentence embeddings
  - Keyword-based search
  - Price range matching
  - Specification-based search
- Ranked search results with similarity scores
- JSON output for easy integration
- Threading for optimized performance
- Cache for query optimization

---

## Requirements
- Python 3.7 or higher
- Required libraries:
  - `sentence-transformers`
  - `torch`
  - `cachetools`
  - `psutil`
  - `numpy`

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/advanced-product-search.git
   cd advanced-product-search
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   python3 -m venv env
   source env/bin/activate   
   # For Windows: .\env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have a valid `products.json` file in the root directory. This file should contain product data in JSON format.

---

## Running the Application

1. **Run the script**:
   ```bash
   python main.py
   ```

2. **Input your query**: When prompted, enter a query such as:
   ```
   Phones under ₹15,000
   ```

3. **Results**: The script will return the top-ranked results as a JSON list, with each product's similarity score.

---

## Input Format for `products.json`

The `products.json` file should contain an array of product objects with the following structure:
```json
[
  {
    "name": "Product Name",
    "description": "Product Description",
    "company": "Company Name",
    "price": {"cost": "12345"},
    "offers": [{"description": "10% Discount"}]
  },
  ...
]
```

---

## Example Usage

### Example Query
```bash
Enter your search query: Phones under ₹15,000
Enter the number of results to display: 5
```

### Example Output
```json
[
    {
        "product": {
            "name": "Phone Model A",
            "description": "Fast processor, 4GB RAM, 64GB Storage",
            "company": "Brand A",
            "price": {"cost": "14000"},
            "offers": [{"description": "Exchange offer available"}]
        },
        "similarity_score": 0.89
    },
    {
        "product": {
            "name": "Phone Model B",
            "description": "High-quality camera, 6GB RAM, 128GB Storage",
            "company": "Brand B",
            "price": {"cost": "15000"},
            "offers": [{"description": "10% cashback"}]
        },
        "similarity_score": 0.84
    }
]
```

---

## Integrating with an LLM for Human-Readable Responses

The JSON results from the script can be passed to a larger LLM (e.g., OpenAI GPT models) for generating human-readable responses. Here's how you can achieve this:

1. **Install OpenAI Library**:
   ```bash
   pip install openai
   ```

2. **Example Python Code**:
   ```python
   import openai
   import json

   # Load search results
   with open('search_results.json', 'r') as f:
       results = json.load(f)

   # Format JSON for LLM input
   product_descriptions = [
       f"Product: {result['product']['name']}\n"
       f"Description: {result['product']['description']}\n"
       f"Price: {result['product']['price']['cost']}\n"
       f"Similarity Score: {result['similarity_score']:.2f}\n"
       for result in results
   ]
   formatted_input = "\n".join(product_descriptions)

   # Generate human-readable response
   openai.api_key = 'your_openai_api_key'
   response = openai.ChatCompletion.create(
       model="gpt-4",
       messages=[
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": f"Summarize the following product search results:\n{formatted_input}"}
       ]
   )

   # Print the response
   print(response['choices'][0]['message']['content'])
   ```

3. **Example LLM Output**:
   ```
   Here are the best matching phones:
   1. Phone Model A: A fast processor with 4GB RAM and 64GB storage, priced at ₹14,000. It comes with an exchange offer.
   2. Phone Model B: A high-quality camera phone with 6GB RAM and 128GB storage, priced at ₹15,000, offering a 10% cashback.
   ```

---

## Extending the System
- **New Matching Strategies**: Add additional matching logic in methods like `_keyword_matching` or `_specification_matching`.
- **Improved JSON Output**: Include additional metadata or filtering options in the output.

---

## Author

Developed by Hariharan Mudaliar. Contributions and suggestions are welcome.

---

## License
This project is licensed under the MIT License.
