import os
import json
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import bert_score
import nltk
from sentence_transformers import SentenceTransformer, util

nltk.download('punkt')      # required for tokenization
nltk.download('punkt_tab')  # dependency for punkt

# Load environment variables
load_dotenv()

# Initialize the Gemini LLM via Vertex AI
llm = ChatVertexAI(
    model_name="gemini-2.5-flash",
    project=os.getenv("GCP_PROJECT_ID"),
    location="us-central1",
    temperature=0.3,   # 0-2  (low for summarization/evaluation)
    max_tokens=2000
)

# Load input JSON
with open("products.json", "r") as f:
    products = json.load(f)

# For embedding similarity (optional)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

results = []

for i, item in enumerate(products[:5], 1):  # taking only 5 to show
    pid = item['product_id']
    description = item['description']
    print("processing product:", pid)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Summarize the following product description:\n\n{description}"}
    ]

    try:
        summary = llm.invoke(messages).content.strip()
    except Exception as e:
        print(f"[ERROR] LLM failed for product {pid}: {e}")
        summary = ""

    # BLEU
    ref_tok = [nltk.word_tokenize(description)]
    gen_tok = nltk.word_tokenize(summary)
    # it has 7 different methods, method1 is most commonly used
    bleu = sentence_bleu(ref_tok, gen_tok, smoothing_function=SmoothingFunction().method1)

    # ROUGE
    # rouge1 for unigrams(one word), rougeL for longest common subsequence
    # use_stemmer does stemming
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge.score(description, summary)

    # BERTScore
    # F1 score is balance between precision and recall
    _, _, F1 = bert_score.score([summary], [description], lang="en", verbose=False)

    # Optional: cosine similarity between embeddings
    # convert_to_tensor means it returns a PyTorch tensor which is multi dimensional array
    desc_emb = embed_model.encode(description, convert_to_tensor=True)
    summ_emb = embed_model.encode(summary, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(desc_emb, summ_emb).item()

    results.append({
        "product_id": pid,
        "description": description,
        "generated_summary": summary,
        "BLEU": round(bleu, 4),
        "ROUGE-1": round(rouge_scores['rouge1'].fmeasure, 4),
        "ROUGE-L": round(rouge_scores['rougeL'].fmeasure, 4),
        "BERTScore-F1": round(F1[0].item(), 4),
        "Cosine-Similarity": round(cosine_sim, 4)
    })

# Save results
with open("summary_evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Evaluation complete. Results saved to summary_evaluation_results.json")