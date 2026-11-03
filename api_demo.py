"""Requests-based smoke/demo for the three assignment endpoints and document workflow."""
import argparse
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", help="optional path to a .txt, .md, .pdf, or .epub file")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    print("GET /", requests.get(base + "/").json())
    print("GET /health", requests.get(base + "/health").json())
    direct = requests.post(base + "/predict", json={"text": "Alice enters the old garden.", "source_language": "English", "target_language": "Japanese", "style_guide": "Preserve names and paragraph structure."})
    direct.raise_for_status()
    print("POST /predict", direct.json())

    if args.path:
        with open(args.path, "rb") as handle:
            document = requests.post(base + "/documents", files={"file": (args.path, handle)})
        document.raise_for_status()
        document = document.json()
        print("POST /documents", document)
        chapter = document["chapters"][0]
        analysis = requests.post(f"{base}/documents/{document['document_id']}/chapters/{chapter['id']}/analyze", json={"source_language": "auto", "target_language": "English"})
        analysis.raise_for_status()
        print("POST chapter analyze", analysis.json())
        translation = requests.post(f"{base}/documents/{document['document_id']}/chapters/{chapter['id']}/translate", json={"source_language": "auto", "target_language": "English", "style_guide": "Preserve paragraph breaks."})
        translation.raise_for_status()
        print("POST chapter translate", translation.json())


if __name__ == "__main__":
    main()
