"""Standalone requests demo: run the server first with `uvicorn app.main:app --reload`."""
import argparse
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to a .txt, .md, .pdf, or .epub file")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    with open(args.path, "rb") as handle:
        document = requests.post(args.base_url + "/documents", files={"file": (args.path, handle)}).json()
    print(document)
    chapter = document["chapters"][0]
    print(requests.post(f"{args.base_url}/documents/{document['document_id']}/chapters/{chapter['id']}/analyze").json())
    print(requests.post(args.base_url + "/predict", json={"document_id": document["document_id"], "chapter_id": chapter["id"]}).json())


if __name__ == "__main__":
    main()
