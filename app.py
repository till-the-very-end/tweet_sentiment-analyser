from flask import Flask, jsonify, request, send_from_directory

from summarizer import rouge_n, summarize

app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.post("/api/summarize")
def api_summarize():
    d = request.get_json(force=True)
    result = summarize(
        d.get("text", ""),
        method=d.get("method", "hybrid"),
        percent=int(d.get("percent", 25)),
        query=d.get("query", ""),
    )
    return jsonify(result)


@app.post("/api/rouge")
def api_rouge():
    d = request.get_json(force=True)
    return jsonify({f"rouge{n}": rouge_n(d["summary"], d["reference"], n) for n in (1, 2)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
