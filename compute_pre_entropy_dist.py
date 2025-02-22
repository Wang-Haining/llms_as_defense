import json
import numpy as np
from pathlib import Path
from roberta import RobertaPredictor
from utils import LogisticRegressionPredictor, SVMPredictor, load_corpus, CORPORA

THREAT_MODELS_DIR = Path("threat_models")
PREDICTOR_CLASSES = {
    "logreg": LogisticRegressionPredictor,
    "svm": SVMPredictor,
    "roberta": RobertaPredictor,
}

def compute_entropy(y_pred_probs: np.ndarray) -> float:
    """Compute mean entropy of predictions."""
    entropies = -np.sum(y_pred_probs * np.log2(y_pred_probs + 1e-10), axis=1)
    return float(np.mean(entropies))

entropy_results = {}

for corpus in CORPORA:
    entropy_results[corpus] = {}
    _, _, test_texts, test_labels = load_corpus(corpus=corpus, task="no_protection")

    for model_type, predictor_class in PREDICTOR_CLASSES.items():
        model_path = THREAT_MODELS_DIR / corpus / "no_protection" / model_type / "model"
        predictor = predictor_class(model_path)

        # get probability predictions
        y_pred_probs = predictor.predict_proba(test_texts)
        entropy = compute_entropy(y_pred_probs)
        entropy_results[corpus][model_type] = entropy

# # save results to JSON
# output_file = Path("results/pre_entropy_dist.json")
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(entropy_results, f, indent=2, ensure_ascii=False)

# print results to stdout
print(json.dumps(entropy_results, indent=2, ensure_ascii=False))
