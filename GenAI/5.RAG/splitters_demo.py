"""
splitters_demo.py  —  All 5 LangChain text splitters in ONE place.

Run:  python splitters_demo.py
Goal: see how differently each method cuts the SAME text.

The ladder (dumb-but-fast  ->  smart-but-costly):
  1. CharacterTextSplitter          cut by ONE separator
  2. RecursiveCharacterTextSplitter cut by a list of separators   <- default
  3. TokenTextSplitter              cut by TOKEN count
  4. MarkdownHeaderTextSplitter     cut by document STRUCTURE (headings)
  5. SemanticChunker                cut by MEANING (uses embeddings)
"""

from dotenv import load_dotenv
load_dotenv()   # <-- reads GCP_PROJECT_ID from your .env (needed for #5)

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
)

# ── Sample text (mixed paragraphs, like our catalogue) ────────────────
SAMPLE = """Returns Policy. Items can be returned within 7 days of delivery if unused and in original packaging. Earbuds are non-returnable once the seal is opened, except for verified defects.

Shipping Policy. Standard delivery is free on orders above 499. Orders below 499 incur a 49 rupee shipping fee. Delivery takes 2 to 5 business days.

Warranty. All electronics carry a 1-year manufacturer warranty. Accessories carry a 6-month warranty."""

# Markdown version (needed for splitter #4, which cuts on # headings)
SAMPLE_MD = """# Returns Policy
Items can be returned within 7 days of delivery if unused.

## Earbuds
Earbuds are non-returnable once the seal is opened.

# Shipping Policy
Standard delivery is free on orders above 499.

# Warranty
All electronics carry a 1-year manufacturer warranty.
"""


def show(title, chunks):
    """Pretty-print the chunks a splitter produced."""
    print(f"\n{'='*60}\n{title}  ->  {len(chunks)} chunks\n{'='*60}")
    for i, c in enumerate(chunks):
        # a chunk may be a plain string OR a Document (has .page_content)
        text = getattr(c, "page_content", c)
        meta = getattr(c, "metadata", "")
        print(f"[chunk {i}] {meta}\n{text}\n")


# ── 1) CharacterTextSplitter ──────────────────────────────────────────
# Cuts on ONE separator (here a blank line). Simplest; can slice mid-sentence.
def demo_character():
    splitter = CharacterTextSplitter(separator="\n\n", chunk_size=200, chunk_overlap=20)
    show("1) CharacterTextSplitter", splitter.split_text(SAMPLE))


# ── 2) RecursiveCharacterTextSplitter (THE DEFAULT) ───────────────────
# Tries separators in order: paragraph -> line -> space -> character,
# so it keeps paragraphs/sentences whole as much as possible.
def demo_recursive():
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    show("2) RecursiveCharacterTextSplitter", splitter.split_text(SAMPLE))


# ── 3) TokenTextSplitter ──────────────────────────────────────────────
# Measures chunk size in TOKENS (not characters). Use to respect a model's
# exact token budget / control cost.
def demo_token():
    try:
        splitter = TokenTextSplitter(chunk_size=40, chunk_overlap=5)
        show("3) TokenTextSplitter", splitter.split_text(SAMPLE))
    except Exception as e:
        print("\n3) TokenTextSplitter -> skipped (needs a one-time tiktoken "
              f"tokenizer download / internet).\nReason: {e}\n")


# ── 4) MarkdownHeaderTextSplitter ─────────────────────────────────────
# Cuts by heading. Attaches the headings as METADATA so each chunk
# remembers its section. It does NOT limit size -> often followed by
# RecursiveCharacterTextSplitter to size-limit long sections.
def demo_markdown():
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")]
    )
    sections = md_splitter.split_text(SAMPLE_MD)          # -> list of Documents
    # optional second step: size-limit each section (keeps the metadata)
    sections = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=20
    ).split_documents(sections)
    show("4) MarkdownHeaderTextSplitter", sections)


# ── 5) SemanticChunker ────────────────────────────────────────────────
# Cuts where the MEANING shifts, using embeddings. Needs an embeddings
# model (costs API calls) -> best for long, dense documents.
#   pip install langchain-experimental
def demo_semantic():
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_google_vertexai import VertexAIEmbeddings
        import os

        embeddings = VertexAIEmbeddings(
            model_name="gemini-embedding-001",
            project=os.getenv("GCP_PROJECT_ID"),
            location="us-central1",
        )
        
        splitter = SemanticChunker(embeddings, 
                                   breakpoint_threshold_type="percentile",
                                #    breakpoint_threshold_amount=0.95 # it is the default value
                                   ) 
        
        show("5) SemanticChunker", splitter.split_text(SAMPLE))
    except Exception as e:
        print(f"\n{'='*60}\n5) SemanticChunker  ->  skipped")
        print(f"{'='*60}\n(needs `pip install langchain-experimental` and Vertex "
              f"credentials)\nReason: {e}\n")


if __name__ == "__main__":
    demo_character()
    demo_recursive()
    demo_token()
    demo_markdown()
    demo_semantic()   # runs only if credentials/lib are available