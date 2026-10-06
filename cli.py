"""Ask questions about a PDF from the terminal: python cli.py path/to/file.pdf"""

import sys
from pathlib import Path

from rag import PdfIndex


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python cli.py path/to/file.pdf")

    print("Indexing...")
    index = PdfIndex.build(Path(sys.argv[1]).read_bytes())
    print(f"Indexed {len(index.chunks)} chunks. Ask a question (empty line to quit).")

    try:
        while question := input("\n> ").strip():
            answer, hits = index.answer(question)
            print(f"\n{answer}")
            print("\nSources: " + ", ".join(f"p. {c.page} ({score:.2f})" for c, score in hits))
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
