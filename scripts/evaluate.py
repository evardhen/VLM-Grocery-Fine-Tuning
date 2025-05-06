from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_score import BERTScorer
from anls import anls_score
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet
import nltk
import torch
from collections import defaultdict

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

def wups_sim():

    nltk.download('wordnet')
    syn1 = wordnet.synsets(sentences[0])[0]
    syn2 = wordnet.synsets(sentences[1])[0]

    print(syn1.wup_similarity(syn2))

def ANLS_sim():
    print(anls_score(prediction=sentences[0], gold_labels=[sentences[1]], threshold=0.5))

def bert_score():
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score([sentences[0]], [sentences[1]])
    print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")

def _extract_categories(text: str, *, fine: bool = False) -> List[str]:
    pattern = RE_FINE if fine else RE_COARSE
    return [m.group(1).strip() for m in pattern.finditer(text)]

def _pairs(text: str) -> list[tuple[str, int]]:
    """
    Return a list of (category, count) tuples, lower‑cased and sorted
    alphabetically by the category string.
    """
    found = [(cat.strip().lower(), int(cnt)) for cat, cnt in PAIR_RE.findall(text)]
    return sorted(found, key=lambda t: t[0])

def _bert_f1(preds: List[str], refs: List[str], scorer: BERTScorer) -> float:
    _, _, f1 = scorer.score(preds, refs)
    return f1.mean().item()




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
        return torch.nn.functional.cosine_similarity(v1, v2, dim=0).item()
    
    def embed_strings(strings):
        """Return a tensor of shape (len(strings), hidden_size)."""
        toks = scorer._tokenizer(
            strings,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(scorer.device)

        with torch.no_grad():
            outs = scorer._model(**toks, output_hidden_states=True, return_dict=True)
            # use the last-layer hidden states exactly like BERTScore does
            hidden = outs.hidden_states[-1]          # (batch, seq, dim)
            mask   = toks.attention_mask.unsqueeze(-1)  # (batch, seq, 1)
            sent_emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # average pooling
            return sent_emb.cpu()
        
    # BERTScore --------------------------------------------------------------
    scorer = BERTScorer(model_type="bert-base-uncased")

    f1_cat = _bert_f1(cat_pred, cat_true, scorer)
    f1_fine = _bert_f1(fine_pred, fine_true, scorer)

    # Exact‑Match for counts --------------------------------------------------
    all_strings = set()
    for i in ids:
        all_strings.update(s for s, _ in _pairs(ground_truth_data[i]["answer"]))
        all_strings.update(s for s, _ in _pairs(pred_data[i]["answer"]))
    all_strings = list(all_strings)                        

    # 2. Embed once
    embeddings = embed_strings(all_strings)  # shape (M, d)
    print("Embedding shape:", embeddings.shape)
    string2vec = dict(zip(all_strings, embeddings))


    # -----------------------------------------------------------------
    # 3. Category‑aware, order‑independent EM
    # -----------------------------------------------------------------
    threshold = 0.60
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

            if best_sim > threshold and ref_cnt == pred_cnt:
                correct += 1
        total += len(ref_pairs)


    count_em = correct / total

    scores = {
        "bert_f1_category": round(f1_cat, 4),
        "bert_f1_fine": round(f1_fine, 4),
        "count_em": round(count_em, 4),
    }

    with open(out_path, "r") as file:
        data = json.load(file)
        data[out_path] = scores

    with open(out_path, "w") as file:
        json.dump(data, file, indent=4)
    
    print(f"Saved scores → {out_path}: {scores}")
    return scores


if __name__ == "__main__":
    # wups_sim()
    # ANLS_sim()
    # bert_score()

    evaluate(ground_truth_path="data/eval.json", pred_path="data/predictions/chat_backend/adapted_inference_image_resolution/qwen2vl_lora_sft_grocery_262144_res.json", out_path="stats/inference_stats/scores.json")