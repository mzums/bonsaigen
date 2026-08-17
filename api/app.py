from flask import Flask
from flask import Response
import random
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = BASE_DIR.parent / "generated"

final_trees = []

for file_path in GENERATED_DIR.rglob("frame_203.txt"):
    print(f"Opening: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        content = f.read()
        final_trees.append(content)


@app.route("/")
def home():
    if not final_trees:
        return "No data found", 404

    return Response(
        f"<pre>{content}</pre>",
        mimetype="text/html"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)