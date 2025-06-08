import json
import pandas as pd
import csv
from neo4j import GraphDatabase
from pathlib import Path


class KnowledgeGraphBuilder:
    """
    Builds a knowledge graph from a product dataset by extracting triples
    and inserting them into a Neo4j graph database.
    """

    def __init__(self, json_file='../data/phone_details.json', triples_file='../data/triples.csv'):
        """
        Initialize file paths and define relationship mappings.

        Args:
            json_file (str): Path to the product details JSON file.
            triples_file (str): Path to output CSV file with extracted triples.
        """
        self.json_file = Path(json_file)
        self.triples_file = Path(triples_file)
        self.important_fields = {
            "manufacturer": "Sản_xuất_bởi",
            "chipset": "Chipset",
            "cpu": "CPU",
            "operating_system": "Hệ_điều_hành",
            "os_version": "Phiên_bản_hệ_điều_hành",
            "display_size": "Kích_thước_màn_hình",
            "display_resolution": "Độ_phân_giải_màn_hình",
            "mobile_type_of_display": "Loại_màn_hình",
            "mobile_tan_so_quet": "Tần_số_quét",
            "camera_primary": "Camera_sau",
            "camera_secondary": "Camera_trước",
            "camera_video": "Video_ghi_hình",
            "mobile_cong_nghe_sac": "Công_nghệ_sạc",
            "sac_khong_day": "Sạc_không_dây",
            "mobile_khang_nuoc_bui": "Kháng_nước_bụi",
            "sim": "Loại_sim",
            "storage": "Bộ_nhớ",
            "mobile_ram_filter": "RAM",
            "mobile_nfc": "NFC",
            "bluetooth": "Bluetooth",
            "wlan": "Wifi",
            "gps": "GPS",
            "dimensions": "Kích_thước",
            "product_weight": "Trọng_lượng",
            "mobile_tinh_nang_dac_biet": "Tính_năng_đặc_biệt",
            "included_accessories": "Phụ_kiện_bao_gồm",
            "warranty_information": "Bảo_hành"
        }

    def extract_triples(self):
        """
        Load product data from JSON, extract relevant triples, and save to CSV.
        """
        if not self.json_file.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_file}")

        with self.json_file.open('r', encoding='utf-8') as f:
            products = json.load(f)

        triples = []

        for product in products:
            general = product.get('general', {})
            attributes = general.get('attributes', {})
            source = general.get('name')

            for attr_key, rel_name in self.important_fields.items():
                value = attributes.get(attr_key)
                if value and value != "no_selection":
                    value_str = str(value).strip()
                    if value_str:
                        triples.append({
                            "head": source.strip(),
                            "relation": rel_name.strip(),
                            "tail": value_str
                        })

        df = pd.DataFrame(triples)
        df.to_csv(self.triples_file, index=False, encoding='utf-8')
        print("✅ Extracted and saved triples to CSV.")
        print(df.head(10))  # Preview sample

    def insert_triples_to_neo4j(self, uri, username, password):
        """
        Insert extracted triples from CSV into a Neo4j graph database.

        Args:
            uri (str): Neo4j URI (e.g., bolt://localhost:7687).
            username (str): Neo4j username.
            password (str): Neo4j password.
        """

        def insert_triple(tx, head, relation, tail):
            """
            Cypher query to insert or merge a triple.
            """
            query = (
                "MERGE (h:Entity {name: $head}) "
                "MERGE (t:Entity {name: $tail}) "
                f"MERGE (h)-[r:`{relation}`]->(t)"
            )
            tx.run(query, head=head, tail=tail)

        if not self.triples_file.exists():
            raise FileNotFoundError(f"CSV triple file not found: {self.triples_file}")

        driver = GraphDatabase.driver(uri, auth=(username, password))

        with driver.session() as session, self.triples_file.open('r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                head = row['head']
                relation = row['relation']
                tail = row['tail']
                print(f"Inserting triple: ({head})-[:{relation}]->({tail})")
                session.execute_write(insert_triple, head, relation, tail)

        driver.close()
        print("✅ All triples inserted into Neo4j successfully.")


if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()

    # Step 1: Extract triples from JSON and save to CSV
    builder.extract_triples()

    # Step 2: Insert into Neo4j
    builder.insert_triples_to_neo4j(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="your_password"  # Change to your real Neo4j password
    )
