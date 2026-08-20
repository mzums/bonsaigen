from flask import Flask
from flask import Response
import random
from pathlib import Path
import html

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

GENERATED_DIR = BASE_DIR.parent / "ml/gpt_conv_enc/generated_frames"
BASE_FILE = BASE_DIR.parent / "get_data/base.txt"

final_trees = []

for file_path in GENERATED_DIR.rglob("*.txt"):
    with file_path.open("r", encoding="utf-8") as f:
        final_trees.append(f.read())

with BASE_FILE.open(encoding="utf-8") as f:
    base = f.read()


@app.route("/")
def home():
    if not final_trees:
        return "No data found", 404

    content = random.choice(final_trees)

    green_colors = [
        "#3E923E",
        "#28D328",
    ]

    brown_colors = [
        "#FD6A01",
        "#A86244",
    ]

    bg = "#242424"

    result = []

    for char in content:
        if char == "&":
            color = random.choice(green_colors)
            result.append(
                f'<span style="color: {color};">&amp;</span>'
            )

        elif char in "/|_\\~.":
            color = random.choice(brown_colors)
            result.append(
                f'<span style="color: {color};">{html.escape(char)}</span>'
            )

        else:
            result.append(html.escape(char))

    content = "".join(result)

    base_escaped = html.escape(base)

    base_escaped = base_escaped.replace(
        "./~~~\\.",
        '<span style="color: #E27120;">./~~~\\.</span>'
    )

    base_escaped = f'<span style="color: white;">{base_escaped}</span>'

    content += "\n" + base_escaped

    return Response(
        f"""
        <html>
            <body style="background-color: {bg}; margin: 0;">
                <pre style="margin: 0;">{content}</pre>
            </body>
        </html>
        """,
        mimetype="text/html"
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002)