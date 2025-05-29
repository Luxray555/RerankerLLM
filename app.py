from flask import Flask, request, jsonify
from sentence_transformers import CrossEncoder

app = Flask(__name__)

# Charger le reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@app.route("/rerank", methods=["POST"])
def rerank():
    data = request.get_json()

    query = data.get("query")
    documents = data.get("documents", [])

    if not query or not documents:
        return jsonify({"error": "Missing 'query' or 'documents'"}), 400

    # Créer les paires (query, doc)
    pairs = [(query, doc) for doc in documents]

    # Prédire les scores
    scores = reranker.predict(pairs)

    # Trier les documents avec leurs scores
    reranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    # Retourner les résultats
    results = [{"document": doc, "score": float(score)} for doc, score in reranked]
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
