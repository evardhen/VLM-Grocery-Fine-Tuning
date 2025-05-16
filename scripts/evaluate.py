from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
import glob
import os

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
#    •  "Category:" or "category:"
RE_COARSE = re.compile(r"(?<!fine[- ]grained\s)category\s*:\s*([^\n]+)", re.IGNORECASE)
#    •  "Fine-grained category:" or "Fine grained Category:" (robust)
RE_FINE = re.compile(r"fine[- ]?grained\s*category\s*:\s*([^\n]+)", re.IGNORECASE)
RE_ANSWER_CLEAN = re.compile(r"[\n\r]+")  # collapse newlines
RE_COUNT = re.compile(r"count\s*[:\-]?\s*(\d+)", re.IGNORECASE)
PAIR_RE = re.compile(
    r"fine[- ]?grained\s*category\s*:\s*([^\n]+?)"     # group 1 = category
    r"(?:\s*[\n\r\-•]*)"                               # delimiters / bullets
    r"count\s*[:\-]?\s*(\d+)",                         # group 2 = count
    re.IGNORECASE | re.DOTALL,
)

sentences = ["onion", "red onion"]

def embedding_sim():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Compute embedding for both lists
    embedding_1 = model.encode(sentences[0], convert_to_tensor=True)
    embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

    sim_score = util.pytorch_cos_sim(embedding_1, embedding_2)
    print(sim_score)

def _extract_categories(text: str, *, fine: bool = False) -> List[str]:
    pattern = RE_FINE if fine else RE_COARSE
    return [m.group(1).strip() for m in pattern.finditer(text)]

def _pairs(text: str) -> list[tuple[str, int]]:
    """
    Return a list of (fine-grained category, count) tuples, lower‑cased and sorted
    alphabetically by the category string.
    """
    found = [(cat.strip().lower(), int(cnt)) for cat, cnt in PAIR_RE.findall(text)]
    return sorted(found, key=lambda t: t[0])


def evaluate(pred_path: str | Path, ground_truth_path: str | Path, out_path: str | Path) -> Dict[str, float]:
    pred_data = {d["question_id"]: d for d in json.load(Path(pred_path).open())}
    ground_truth_data = {d["question_id"]: d for d in json.load(Path(ground_truth_path).open())}

    ids = sorted(set(pred_data) & set(ground_truth_data))
    if not ids:
        raise ValueError("No overlapping question_ids between files.")
    print(f"Matched {len(ids)} samples.")

    # Regex filtering
    cat_pred = ["; ".join(_extract_categories(pred_data[i]["answer"], fine=False)) or "" for i in ids]
    cat_true = ["; ".join(_extract_categories(ground_truth_data[i]["answer"], fine=False)) or "" for i in ids]

    fine_pred = ["; ".join(_extract_categories(pred_data[i]["answer"], fine=True)) or "" for i in ids]
    fine_true = ["; ".join(_extract_categories(ground_truth_data[i]["answer"], fine=True)) or "" for i in ids]

    # Helper functions
    def _sim(a: str, b: str) -> float:
        v1, v2 = string2vec[a], string2vec[b]
        return model.similarity(v1, v2)
     
    # BERTScore --------------------------------------------------------------
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    cat_pred_embed = model.encode(cat_pred)
    cat_true_embed = model.encode(cat_true)
    print(cat_true_embed.shape)
    fine_pred_embed = model.encode(fine_pred)
    fine_true_embed = model.encode(fine_true)
    cos_cat = model.similarity(cat_pred_embed, cat_true_embed).diagonal().mean()
    cos_fine = model.similarity(fine_pred_embed, fine_true_embed).diagonal().mean()


    # Exact‑Match for counts --------------------------------------------------
    all_strings = set()
    for i in ids:
        all_strings.update(s for s, _ in _pairs(ground_truth_data[i]["answer"]))
        all_strings.update(s for s, _ in _pairs(pred_data[i]["answer"]))
    all_strings = list(all_strings)                        

    # 2. Embed once
    embeddings= model.encode(all_strings, convert_to_tensor=True)
    # embeddings = embed_strings(all_strings)  # shape (M, d)
    string2vec = dict(zip(all_strings, embeddings))


    # -----------------------------------------------------------------
    # 3. Category‑aware, order‑independent EM
    # -----------------------------------------------------------------
    threshold = 0.50
    correct, total = 0, 0

    for i in ids:
        ref_pairs  = _pairs(ground_truth_data[i]["answer"])
        pred_pairs = _pairs(pred_data[i]["answer"])

        matched = set()

        # greedy assignment: each ref chooses the most similar unused pred
        for ref_cat, ref_cnt in ref_pairs:
            sims = [
                (j, _sim(ref_cat, pc)) for j, (pc, _) in enumerate(pred_pairs)
                if j not in matched
            ]
            if not sims:
                break

            j_best, best_sim = max(sims, key=lambda x: x[1])
            _, pred_cnt = pred_pairs[j_best]
            matched.add(j_best)

            if best_sim > threshold:
                total += 1
            if best_sim > threshold and ref_cnt == pred_cnt:
                correct += 1

    if total == 0:
        count_em = 0.0
    else:
        count_em = correct / total

    scores = {
        "bert_cos_category": round(float(cos_cat), 4),
        "bert_cos_fine": round(float(cos_fine), 4),
        "count_em": round(count_em, 4),
    }

    with open(out_path, "r") as file:
        data = json.load(file)
        data[pred_path] = scores

    with open(out_path, "w") as file:
        json.dump(data, file, indent=4)
    
    print(f"Saved scores → {out_path}: {scores}")
    return scores


if __name__ == "__main__":
    predictions_path = "data/predictions/chat_backend/fixed_inference_image_resolution_589824"

    for path in glob.glob(os.path.join(predictions_path, "*base.json")):
        evaluate(ground_truth_path="data/eval.json", pred_path=path, out_path="stats/inference_stats/scores.json")

    # evaluate(ground_truth_path="data/eval.json", pred_path="data/predictions/openai/openai_gpt4o_predictions.json", out_path="stats/inference_stats/scores.json")