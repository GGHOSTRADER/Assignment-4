# query_system.py
"""Minimal KG query template for Assignment 4.

Keep these APIs unchanged for auto-test:
- generate_text(messages, max_new_tokens=220)
- get_relevant_articles(question)
- generate_answer(question, rule_results)

Keep Rule fields aligned with build_kg output:
rule_id, type, action, result, art_ref, reg_name
"""

import os
from typing import Any

from neo4j import GraphDatabase
from dotenv import load_dotenv

from llm_loader import load_local_llm, get_tokenizer, get_raw_pipeline


# ========== 0) Initialization ==========
load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (
    os.getenv("NEO4J_USER", "neo4j"),
    os.getenv("NEO4J_PASSWORD", "password"),
)

# Avoid local proxy settings interfering with model/Neo4j access.
for key in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    if key in os.environ:
        del os.environ[key]


try:
    driver = GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()
except Exception as e:
    print(f"⚠️ Neo4j connection warning: {e}")
    driver = None


# ========== 1) Public API (query flow order) ==========
# Order: extract_entities -> build_typed_cypher -> get_relevant_articles -> generate_answer


def generate_text(messages: list[dict[str, str]], max_new_tokens: int = 220) -> str:
    """
    Call local HF model via chat template + raw pipeline.

    Interface:
    - Input:
      - messages: list[dict[str, str]] (chat messages with role/content)
      - max_new_tokens: int
    - Output:
      - str (model generated text, no JSON guarantee)
    """
    tok = get_tokenizer()
    pipe = get_raw_pipeline()
    if tok is None or pipe is None:
        load_local_llm()
        tok = get_tokenizer()
        pipe = get_raw_pipeline()
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return pipe(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"].strip()


def extract_entities(question: str) -> dict[str, Any]:
    """TODO(student, required): parse question to {question_type, subject_terms, aspect}."""
    return {
        "question_type": "general",
        "subject_terms": [],
        "aspect": "general",
    }


def build_typed_cypher(entities: dict[str, Any]) -> tuple[str, str]:
    """TODO(student, required): return (typed_query, broad_query) with score and required fields."""
    cypher_typed = """
	"""

    cypher_broad = """
	"""

    return cypher_typed, cypher_broad


def get_relevant_articles(question: str) -> list[dict[str, Any]]:
    """TODO(student, required): run typed+broad retrieval and return merged rule dicts."""
    if driver is None:
        return []
    return []


def generate_answer(question: str, rule_results: list[dict[str, Any]]) -> str:
    """TODO(student, required): generate grounded answer from retrieved rules only."""
    return "Insufficient rule evidence to answer this question."


def main() -> None:
    """Interactive CLI (provided scaffold)."""
    if driver is None:
        return

    load_local_llm()

    print("=" * 50)
    print("🎓 NCU Regulation Assistant (Template)")
    print("=" * 50)
    print("💡 Try: 'What is the penalty for forgetting student ID?'")
    print("👉 Type 'exit' to quit.\n")

    while True:
        try:
            user_q = input("\nUser: ").strip()
            if not user_q:
                continue
            if user_q.lower() in {"exit", "quit"}:
                print("👋 Bye!")
                break

            results = get_relevant_articles(user_q)
            answer = generate_answer(user_q, results)
            print(f"Bot: {answer}")

        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break
        except NotImplementedError as e:
            print(f"⚠️ {e}")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    driver.close()


if __name__ == "__main__":
    main()
