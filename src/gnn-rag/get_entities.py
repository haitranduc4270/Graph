import os
import pandas as pd
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# Gemini API key for Google Generative Language API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

def parse_output_text(output_text):
    """
    Parse the raw text output from Gemini API into structured entities and relationships.

    Args:
        output_text (str): Raw response from Gemini model.

    Returns:
        tuple: A tuple (entities, relationships) where:
            - entities is a list of dictionaries { "entity": ..., "type": ... }
            - relationships is a list of dictionaries { "from": ..., "relation": ..., "to": ... }
    """
    entities = []
    relationships = []
    lines = output_text.strip().split('\n')
    section = None

    for line in lines:
        line = line.strip()
        if line.startswith("Entities:"):
            section = "entities"
            continue
        elif line.startswith("Relationships:"):
            section = "relationships"
            continue
        elif line.startswith("-"):
            if section == "entities":
                # Format: - {Entity}: {Type}
                parts = line.strip("- ").split(":")
                if len(parts) == 2:
                    entity = parts[0].strip(" {}")
                    entity_type = parts[1].strip(" {}")
                    entities.append({"entity": entity, "type": entity_type})
            elif section == "relationships":
                # Format: - (Entity1, Relationship, Entity2)
                content = line.strip("- ()")
                parts = [p.strip() for p in content.split(",")]
                if len(parts) == 3:
                    relationships.append({
                        "from": parts[0],
                        "relation": parts[1],
                        "to": parts[2]
                    })
    return entities, relationships

def entities_detection(text_input):
    """
    Send a question to the Gemini API to extract entities and their relationships.

    Args:
        text_input (str): The input question in Vietnamese.

    Returns:
        tuple: (entities, relationships) as extracted from the Gemini API response.
    """
    prompt = (
        "Extract entities (nodes) and their relationships (edges) from the text below.\n"
        "Entities and relationships MUST be in Vietnamese\n"
        "Valid relations: ['Sản xuất bởi', 'Chipset', 'CPU', 'Hệ điều hành', 'Phiên bản hệ điều hành', "
        "'Kích thước màn hình', 'Độ phân giải màn hình', 'Loại màn hình', 'Tần số quét', 'Camera sau', "
        "'Camera trước', 'Video ghi hình', 'Công nghệ sạc', 'Sạc không dây', 'Kháng nước bụi', 'Loại sim', "
        "'Bộ nhớ', 'RAM', 'NFC', 'Bluetooth', 'Wifi', 'GPS', 'Kích thước', 'Trọng lượng', 'Tính năng đặc biệt', "
        "'Phụ kiện bao gồm', 'Bảo hành']\n\n"
        "Follow this format: \n"
        "Entities:\n- {Entity}: {Type}\n\n"
        "Relationships:\n- ({Entity1}, {RelationshipType}, {Entity2})\n\n"
        f'Text:\n"{text_input}"\n\n'
        "Output:\nEntities:\n- {Entity}: {Type}\n...\n\nRelationships:\n- ({Entity1}, {RelationshipType}, {Entity2})"
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
        entities, relationships = parse_output_text(output_text)
        return entities, relationships
    else:
        print(f"API Error: {response.status_code}")
        print(response.text)
        return [], []

def process_batch(batch):
    """
    Process a batch of QA items to extract entities and relationships from questions.

    Args:
        batch (list): List of dictionaries with keys "question" and "answer".

    Returns:
        list: List of processed results with extracted entities and relationships.
    """
    results = []
    for item in batch:
        query = item["question"]
        expected_answer = item["answer"]
        entities, relationships = entities_detection(query)
        entity_names = [e["entity"] for e in entities]

        results.append({
            "question": query,
            "expected_answer": expected_answer,
            "relations": relationships,
            "entities": entity_names
        })
    return results

def main():
    """
    Main function to run the pipeline:
    - Load dataset
    - Split into batches
    - Process each batch in parallel
    - Save the results to a JSON file
    """
    # Load CSV file (replace path if needed)
    df = pd.read_csv("../../data/qa_sample_500.csv")
    df = df.head(1)

    # Convert to list of dictionaries
    qa_list = df[["question", "answer"]].to_dict(orient="records")

    # Create batches
    batch_size = 50
    batches = [qa_list[i:i + batch_size] for i in range(0, len(qa_list), batch_size)]

    all_results = []

    # Run batch processing in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]
        for future in as_completed(futures):
            all_results.extend(future.result())

    # Save result to JSON file
    with open("qa_with_entities.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
