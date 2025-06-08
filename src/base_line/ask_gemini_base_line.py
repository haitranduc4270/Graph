import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()

# =========================
# Configuration
# =========================

# Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")  # Use env or fallback
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# =========================
# Gemini API Call
# =========================

def ask_gemini(retriever_data, query):
    """
    Sends a prompt to the Gemini API with retrieved context and returns the generated answer.
    
    Args:
        retriever_data (str): The retrieved data (usually Neo4j query results) as input context.
        query (str): The user's natural language question.

    Returns:
        str: Generated answer from Gemini.
    """
    prompt = (
        f"Query:\n\"{query}\"\n\n"
        f"Retriever data:\n{"\n".join(retriever_data)}\n\n"
        f"Answer:"
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        result = response.json()
        output_text = result['candidates'][0]['content']['parts'][0]['text']
        return output_text
    else:
        print(f"[ERROR] Gemini API call failed with status code: {response.status_code}")
        return ''

# =========================
# Cypher Query Execution
# =========================

def run_cypher(query: str, params: dict | None = None):
    """
    Executes a Cypher query on the Neo4j database.

    Args:
        query (str): Cypher query string.
        params (dict, optional): Query parameters.

    Returns:
        list: List of result records in dictionary format.
    """
    try:
        with driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]
    except Exception as e:
        print(f"[ERROR] Failed Cypher query:\n{query}")
        print(f"[DETAIL] Exception: {e}")
        return []

# =========================
# Q&A Pipeline Processing
# =========================

def process_question(question: str, cypher_query: str) -> str:
    """
    Given a natural language question and its corresponding Cypher query, retrieve data from Neo4j
    and ask Gemini to generate an answer.

    Args:
        question (str): User's natural language question.
        cypher_query (str): Corresponding Cypher query.

    Returns:
        str: Generated answer.
    """
    record_data = ''
    if cypher_query != "0":
        records = run_cypher(cypher_query)
        record_data = json.dumps(records, ensure_ascii=False, indent=2)

    answer = ask_gemini(record_data, question)
    return answer

# =========================
# Batch Processing
# =========================

def process_single(index: int, question: str, expected_answer: str, cypher_query: str) -> dict:
    """
    Processes a single question-answer pair.

    Args:
        index (int): Index of the row.
        question (str): User's question.
        expected_answer (str): Ground truth answer.
        cypher_query (str): Corresponding Cypher query.

    Returns:
        dict: Result dictionary with question, expected answer, and generated answer.
    """
    generated_answer = process_question(question, cypher_query)

    # Ensure everything is string for consistency
    generated_answer = str(generated_answer)
    expected_answer = str(expected_answer)

    print("Question         :", question)
    print("Cypher Query     :", cypher_query)
    print("Generated Answer :", generated_answer)

    return {
        "question": question,
        "expected_answer": expected_answer,
        "generated_answer": generated_answer,
    }

# =========================
# Main Execution
# =========================

def main():
    """
    Main pipeline: Reads questions from CSV, processes them in parallel, and saves results.
    """
    input_file = "qa_sample_500_with_cypher.csv"
    output_file = "evaluation_results.csv"
    batch_size = 20

    # Load CSV data
    df = pd.read_csv(input_file)
    num_rows = len(df)
    results = []

    # Process in batches using multi-threading
    for start_idx in range(0, num_rows, batch_size):
        end_idx = min(start_idx + batch_size, num_rows)
        batch = df.iloc[start_idx:end_idx]

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {
                executor.submit(process_single, idx, row["question"], row["answer"], row["cypher_query"]): idx
                for idx, row in batch.iterrows()
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    idx = futures[future]
                    print(f"[ERROR] Processing index {idx} failed: {e}")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to '{output_file}'")

# Run the script
if __name__ == "__main__":
    main()
