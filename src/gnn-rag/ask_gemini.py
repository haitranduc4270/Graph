import json
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Replace this with your actual Gemini API key
GEMINI_API_KEY = ""


class GeminiAPIClient:
    """
    Client to interact with Gemini API for generating content from a prompt.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        self.headers = {"Content-Type": "application/json"}

    def generate_answer(self, query: str, retriever_data: str) -> str:
        """
        Sends a request to Gemini API with the constructed prompt and returns the generated answer.
        Args:
            query (str): The user query.
            retriever_data (str): Retrieved context to be used in prompt.

        Returns:
            str: Generated answer from Gemini or empty string if failed.
        """
        prompt = (
            f"Query:\n\"{query}\"\n\n"
            f"Retriever data:\n{retriever_data}\n\n"
            f"Answer:"
        )

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ]
        }

        response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload))

        if response.status_code == 200:
            result = response.json()
            try:
                return result['candidates'][0]['content']['parts'][0]['text']
            except (KeyError, IndexError):
                print("Unexpected response format.")
                return ""
        else:
            print(f"Request failed with status code {response.status_code}")
            print(response.text)
            return ""


class QAProcessor:
    """
    Processes QA evaluation pipeline using Gemini API to generate answers and compare them to ground truth.
    """

    def __init__(self, input_csv: str, output_csv: str, api_key: str, batch_size: int = 50):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.batch_size = batch_size
        self.api_client = GeminiAPIClient(api_key)
        self.df = pd.read_csv(input_csv)
        self.results = []

    def _process_single(self, idx: int, question: str, expected_answer: str, retriever_data: str) -> dict:
        """
        Processes a single question and returns a dictionary of results.
        """
        generated_answer = self.api_client.generate_answer(question, retriever_data)
        generated_answer = str(generated_answer)  # Ensure string type
        expected_answer = str(expected_answer)

        print(f"Question          : {question}")
        print(f"Generated Answer  : {generated_answer}")

        return {
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer
        }

    def run(self):
        """
        Runs the QA evaluation in batches and writes results to the output CSV.
        """
        num_rows = len(self.df)

        for start_idx in range(0, num_rows, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_rows)
            batch = self.df.iloc[start_idx:end_idx]

            with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
                futures = {
                    executor.submit(
                        self._process_single,
                        idx,
                        row["query"],
                        row["expected_answer"],
                        row["retriever_data"]
                    ): idx
                    for idx, row in batch.iterrows()
                }

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        self.results.append(result)
                    except Exception as e:
                        idx = futures[future]
                        print(f"Error processing index {idx}: {e}")

            time.sleep(1)  # Throttle to avoid API rate limit

        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.output_csv, index=False)
        print(f"\nEvaluation completed. Results saved to {self.output_csv}")


if __name__ == "__main__":
    # Initialize and run the QA evaluation pipeline
    qa_processor = QAProcessor(
        input_csv="qa_retriever_results.csv",
        output_csv="evaluation_results.csv",
        api_key=GEMINI_API_KEY,
        batch_size=50
    )
    qa_processor.run()
