import pandas as pd
import nltk
from evaluate import load

# Ensure required NLTK resources are downloaded
nltk.download('punkt')

class TextEvaluation:
    """
    A class for evaluating generated text against expected answers using common NLP metrics.
    
    Attributes:
        rouge (evaluate.EvaluationModule): Module to compute ROUGE scores.
        meteor (evaluate.EvaluationModule): Module to compute METEOR scores.
        predictions (List[str]): List of generated answers.
        references (List[str]): List of reference answers.
    """

    def __init__(self, csv_path: str):
        """
        Initialize the TextEvaluation class with the path to the CSV file.

        Args:
            csv_path (str): Path to the evaluation results CSV file.
        """
        self.rouge = load("rouge")
        self.meteor = load("meteor")
        self.predictions = []
        self.references = []
        self._load_data(csv_path)

    def _load_data(self, csv_path: str):
        """
        Load and preprocess the CSV data, filtering out missing values.

        Args:
            csv_path (str): Path to the evaluation results CSV file.
        """
        df = pd.read_csv(csv_path)
        df_valid = df[df["generated_answer"].notna()]
        self.predictions = df_valid["generated_answer"].tolist()
        self.references = df_valid["expected_answer"].tolist()

    def compute_scores(self):
        """
        Compute ROUGE and METEOR scores based on loaded predictions and references.

        Returns:
            dict: A dictionary containing the average ROUGE-1, ROUGE-2, ROUGE-L, and METEOR scores.
        """
        rouge_scores = self.rouge.compute(predictions=self.predictions, references=self.references)
        meteor_scores = self.meteor.compute(predictions=self.predictions, references=self.references)

        return {
            "ROUGE-1": rouge_scores["rouge1"],
            "ROUGE-2": rouge_scores["rouge2"],
            "ROUGE-L": rouge_scores["rougeL"],
            "METEOR": meteor_scores["meteor"]
        }

    def print_scores(self):
        """
        Compute and print the evaluation scores in a formatted manner.
        """
        scores = self.compute_scores()
        print(f"ROUGE-1: {scores['ROUGE-1']:.4f}")
        print(f"ROUGE-2: {scores['ROUGE-2']:.4f}")
        print(f"ROUGE-L: {scores['ROUGE-L']:.4f}")
        print(f"METEOR:  {scores['METEOR']:.4f}")


if __name__ == "__main__":
    # Instantiate the evaluator with the path to your CSV file
    evaluator = TextEvaluation("evaluation_results.csv")
    
    # Print out the evaluation metrics
    evaluator.print_scores()
