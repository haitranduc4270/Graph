import os
import time
import json
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load GEMINI API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Load entities and relations from CSV files
entities_df = pd.read_csv("../../data/entities.csv")
relations_df = pd.read_csv("../../data/relations.csv")

# Convert entity and relation names to lists of strings
entities = entities_df['entity'].astype(str).tolist()
relations = relations_df['relation'].astype(str).tolist()

# Cypher generation prompt template used for LLM request
CYPHER_GENERATION_TEMPLATE = """
Task: Generate a Cypher statement to query a graph database.
Instructions:
- Analyze the question and extract relevant graph components dynamically. Use this to construct the Cypher query.
- Use only the relationship types and properties from the provided schema. Do not include any other relationship types.
- The schema is based on a graph structure with nodes and relationships as follows:
relations: {schema}
entities: {entities}
- Return only the generated Cypher query in your response. Do not include explanations, comments, or additional text.
- Ensure the Cypher query directly addresses the given question using the schema accurately.
- If you can't generate it, return "0"
The question is:
{question}
"""

def generate_cypher_query(question: str) -> str:
    """
    Generate a Cypher query from a natural language question using the Gemini API.

    Args:
        question (str): A natural language question.

    Returns:
        str: A generated Cypher query, or '0' if generation failed.
    """
    print("Processing question:", question)

    # Format the LLM prompt using the question, entity, and relation schema
    prompt = CYPHER_GENERATION_TEMPLATE.format(
        question=question,
        schema=', '.join(relations),
        entities=', '.join(entities)
    )

    # Send the request to the Gemini API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Parse and return the result
    if response.status_code == 200:
        result = response.json()
        output_text = result['candidates'][0]['content']['parts'][0]['text']
        return output_text.strip()
    else:
        print(f"Error: API returned status code {response.status_code}")
        print(response.text)
        return '0'

def load_questions(file_path: str) -> pd.DataFrame:
    """
    Load question-answer dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with 'question' and 'answer' columns.
    """
    try:
        df = pd.read_csv(file_path)
        if "question" not in df.columns or "answer" not in df.columns:
            raise ValueError("CSV must contain 'question' and 'answer' columns")
        return df
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        exit(1)
    except ValueError as ve:
        print("Error:", ve)
        exit(1)

def process_questions(df: pd.DataFrame, output_file: str, batch_size: int = 20):
    """
    Generate Cypher queries for a batch of questions and save the results to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing questions and answers.
        output_file (str): Path to the output CSV file.
        batch_size (int): Number of rows to write per batch.
    """
    batch_data = []

    for idx, (question, answer) in enumerate(zip(df['question'], df['answer']), start=1):
        print(f"\n[Question {idx}]")
        print(question)

        cypher = generate_cypher_query(question)
        print("Generated Cypher Query:", cypher)

        batch_data.append({'question': question, 'cypher_query': cypher, 'answer': answer})
        time.sleep(0.1)  # Rate limiting

        # Write in batches
        if idx % batch_size == 0:
            batch_df = pd.DataFrame(batch_data)
            batch_df.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))
            batch_data = []

    # Write any remaining data
    if batch_data:
        batch_df = pd.DataFrame(batch_data)
        batch_df.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))

    print("\n✅ File saved successfully with Cypher queries.")

if __name__ == "__main__":
    input_file_path = "../../data/qa_sample_500.csv"
    output_file_path = "qa_sample_500_with_cypher.csv"

    qa_df = load_questions(input_file_path)
    process_questions(qa_df, output_file_path)
