from flask import Flask, request, jsonify
from bleurt import score

app = Flask(__name__)
checkpoint = "./"
scorer = score.BleurtScorer(checkpoint)


@app.route('/score', methods=['POST'])
def get_bleurt_score():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    references = data.get("references", [])
    candidates = data.get("candidates", [])
    if not references or not candidates:
        return jsonify({"error": "References and candidates are required"}), 400
    scores = scorer.score(references=references, candidates=candidates)
    return jsonify(scores)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
