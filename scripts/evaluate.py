from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_score import BERTScorer
from anls import anls_score
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet
import nltk


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

    # BERTScore calculation
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score([sentences[0]], [sentences[1]])
    print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")

if __name__ == "__main__":
    # wups_sim()
    # ANLS_sim()
    bert_score()