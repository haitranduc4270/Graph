import requests
import json
import time
import os
from pathlib import Path


class ProductDetailFetcher:
    """
    A class to fetch detailed product data from the Cellphones API,
    process products in chunks, and save the results to a JSON file.
    """

    def __init__(self, input_file='../data/phones_all.json', output_file='../data/phone_details.json'):
        """
        Initializes file paths and request headers.

        Args:
            input_file (str): Path to the input JSON file containing product IDs.
            output_file (str): Path to save the fetched product detail results.
        """
        self.input_path = Path(input_file)
        self.output_path = Path(output_file)
        self.chunk_size = 50
        self.api_url = "https://api.cellphones.com.vn/v2/graphql/query"
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "origin": "https://cellphones.com.vn",
            "referer": "https://cellphones.com.vn/",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

    def delay(self, seconds):
        """
        Pause execution for a given number of seconds.

        Args:
            seconds (int): Duration to sleep.
        """
        time.sleep(seconds)

    def get_product_detail(self, product_id):
        """
        Sends a GraphQL request to fetch detailed product data by ID.

        Args:
            product_id (int): The ID of the product to fetch.

        Returns:
            dict or None: Product details as a dictionary, or None if request fails.
        """
        query = f"""
        query getProductDataDetail {{
          product(
            id: {product_id},
            provinceId: 30
          ) {{
            general {{
              name
              attributes
              product_id
              categories {{
                name
                uri
              }}
              review {{
                total_count
                average_rating
              }}
            }}
            filterable {{
              promotion_pack
              price
              prices
              warranty_information
            }}
          }}
        }}
        """

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"query": query, "variables": {}}
            )
            response.raise_for_status()
            return response.json().get('data', {}).get('product', None)
        except requests.RequestException as e:
            print(f"Error fetching productId={product_id}: {e}")
            return None

    def process_chunk(self, products, start_index):
        """
        Process a chunk of the product list.

        Args:
            products (list): The full list of product entries.
            start_index (int): Starting index for the current chunk.

        Returns:
            list: List of detailed product entries fetched.
        """
        chunk = products[start_index:start_index + self.chunk_size]
        results = []

        for product in chunk:
            product_id = product.get('general', {}).get('product_id')
            if not product_id:
                continue
            print(f"Fetching product details for ID: {product_id}")
            detail = self.get_product_detail(product_id)
            if detail:
                results.append(detail)

        return results

    def run(self):
        """
        Main execution method: reads product list, processes in chunks, and saves output.
        """
        if not self.input_path.exists():
            print("Input file phones_all.json does not exist.")
            return

        with self.input_path.open(encoding='utf-8') as f:
            products = json.load(f)

        total_chunks = (len(products) + self.chunk_size - 1) // self.chunk_size
        all_results = []

        for i in range(total_chunks):
            print(f"\nProcessing chunk {i + 1}/{total_chunks}")
            chunk_results = self.process_chunk(products, i * self.chunk_size)
            all_results.extend(chunk_results)

            # Save interim results after each chunk
            with self.output_path.open("w", encoding='utf-8') as f_out:
                json.dump(all_results, f_out, indent=2, ensure_ascii=False)

            print(f"Saved {len(all_results)} products to file.")

            # Wait 10 seconds between chunks (except after the last one)
            if i < total_chunks - 1:
                print("Waiting 10 seconds before next chunk...")
                self.delay(10)

        print("\n✅ Product detail fetching completed!")
        print(f"🟢 Total products saved: {len(all_results)}")


if __name__ == "__main__":
    fetcher = ProductDetailFetcher()
    fetcher.run()
