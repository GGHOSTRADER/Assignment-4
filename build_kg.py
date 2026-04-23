# build_kg.py
"""Minimal KG builder template for Assignment 4.

Keep this contract unchanged:
- Graph: (Regulation)-[:HAS_ARTICLE]->(Article)-[:CONTAINS_RULE]->(Rule)
- Article: number, content, reg_name, category
- Rule: rule_id, type, action, result, art_ref, reg_name
- Fulltext indexes: article_content_idx, rule_idx
- SQLite file: ncu_regulations.db
"""

import os
import sqlite3
from typing import List, Dict, Tuple, Iterable, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase
import hashlib
import json
import re

from llm_loader import load_local_llm, get_tokenizer, get_raw_pipeline


# At the top, add:
from sentence_transformers import SentenceTransformer


# ========== 0) Initialization ==========
load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (
    os.getenv("NEO4J_USER", "neo4j"),
    os.getenv("NEO4J_PASSWORD", "password"),
)

BATCH_SIZE = 3

_embedder: SentenceTransformer | None = None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def chunked(seq: List[Tuple], size: int) -> Iterable[List[Tuple]]:
    """Yield successive chunks from a list."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# ========== 1) Reg-name → type mapping ==========

# Keyword fragments matched (via regex) against the lowercased regulation name.
# Order matters: more specific entries first.
_REG_NAME_TYPE_RULES: List[Tuple[List[str], str]] = [
    # These map regulation names to the dominant rule flavour found in that regulation.
    # The LLM still classifies each sentence — this is used as the default/fallback type
    # when the LLM fails, and as a hint in the prompt.
    (["exam", "examination", "test rules", "invigilation"], "obligation"),
    (["grade", "grading", "gpa", "academic performance", "score"], "procedure"),
    (
        ["credit recognition", "credit waiver", "credit transfer", "credit exemption"],
        "procedure",
    ),
    (
        [
            "course selection",
            "course enrollment",
            "add.*drop",
            "course registration",
            "curriculum",
        ],
        "procedure",
    ),
    (["student id", "identification card", "id card", "student card"], "procedure"),
    # broad/master regulations — intentionally last
    (
        [
            "academic regulations",
            "student regulations",
            "study regulations",
            "general regulations",
            "academic affairs",
            "undergraduate",
            "postgraduate",
            "enrollment regulations",
            "registration regulations",
        ],
        "obligation",
    ),
]


def reg_name_to_type(reg_name: str) -> Optional[str]:
    """
    Derive the rule type deterministically from the regulation name.
    Returns None if no pattern matches.
    """
    low = reg_name.lower()
    for keywords, rule_type in _REG_NAME_TYPE_RULES:
        for kw in keywords:
            if re.search(kw, low):
                return rule_type
    return None


def resolve_type(reg_name: str) -> str:
    """Always returns a valid type; falls back to 'general' if unmatched."""
    return reg_name_to_type(reg_name) or "general"


# ========== 2) Prompt — LLM only fills action + result ==========


def build_prompt(article_number: str, reg_name: str, content: str) -> str:
    """
    Constructs a structured prompt for Qwen-2.5-3B to extract verbatim rules
    from regulation text into JSON format.
    """
    tokenizer = get_tokenizer()

    # System message defining the extraction logic and strict verbatim requirements
    messages = [
        {
            "role": "system",
            "content": (
                "You are a legal data extractor. Your task is to extract rules from regulation text "
                "into a structured JSON format. You must remain 100% faithful to the original wording.\n\n"
                "FIELDS:\n"
                "- type: EXACTLY one of: [obligation, prohibition, permission, penalty, procedure, other]\n"
                "- action: The specific requirement, condition, or act (Verbatim from text).\n"
                "- result: The specific consequence, outcome, or punishment (Verbatim from text). "
                'If no consequence is mentioned, use "".\n\n'
                "TYPE DEFINITIONS:\n"
                "- obligation: Something that MUST or SHALL be done.\n"
                "- prohibition: Something that MUST NOT or MAY NOT be done.\n"
                "- permission: Something that MAY or IS ALLOWED.\n"
                "- penalty: A punishment or specific consequence for a violation.\n"
                "- procedure: Processes, deadlines, or application steps.\n"
                "- other: Definitions or general scope statements.\n\n"
                "STRICT RULES:\n"
                "1. VERBATIM ONLY: Use the EXACT words from the original text. Do not paraphrase or summarize.\n"
                "2. ATOMIC RULES: If a sentence contains two distinct requirements, create two separate rule objects.\n"
                "3. WHITESPACE: Remove mid-sentence line breaks (\\n) but keep all punctuation and words.\n"
                '4. NO INVENTIONS: If no clear rule exists, return {"rules": []}.\n\n'
                "EXAMPLE:\n"
                'Text: "Students must bring their ID to exams. Failure to do so will result in a 5-point deduction."\n'
                "Output:\n"
                "{\n"
                '  "rules": [\n'
                "    {\n"
                '      "type": "obligation",\n'
                '      "action": "Students must bring their ID to exams.",\n'
                '      "result": "Failure to do so will result in a 5-point deduction."\n'
                "    }\n"
                "  ]\n"
                "}\n\n"
                "Return ONLY valid JSON."
            ),
        },
        {
            "role": "user",
            "content": f"Regulation: {reg_name}\nArticle: {article_number}\n\nText:\n{content}",
        },
    ]

    # apply_chat_template formats the messages with the specific Qwen-2.5 tokens
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_llm_response(response_text: str) -> Dict:
    """Parse a model response into the required JSON structure."""
    try:
        return json.loads(response_text)
    except Exception:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return {"rules": []}


def extract_entities_batch(batch_articles: List[Tuple[str, str, str]]) -> List[Dict]:
    """
    Batch extract rules for a list of:
    (article_number, reg_name, content)
    """
    llm = get_raw_pipeline()
    prompts = [
        build_prompt(article_number, reg_name, content)
        for article_number, reg_name, content in batch_articles
    ]

    responses = llm(prompts, batch_size=BATCH_SIZE)

    parsed_outputs: List[Dict] = []
    for response in responses:
        # HF pipeline may return dict or list[dict] depending on version/config
        if isinstance(response, list):
            response_text = response[0].get("generated_text", "")
        else:
            response_text = response.get("generated_text", "")
        parsed_outputs.append(parse_llm_response(response_text))

    return parsed_outputs


def normalize_text(text: str) -> str:
    """Normalize text for deduplication and stable IDs."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def deduplicate_rules(rules: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    unique_rules = []

    for rule in rules:
        key = (
            normalize_text(rule.get("type", "")),
            normalize_text(rule.get("action", "")),
            normalize_text(rule.get("result", "")),
        )

        if not rule.get("action"):
            continue

        if key in seen:
            continue

        seen.add(key)
        unique_rules.append(rule)

    return unique_rules


def make_rule_id(article_number: str, reg_name: str, action: str, result: str) -> str:
    """
    Build a stable unique rule ID from article context + normalized rule text.
    """
    base = "||".join(
        [
            normalize_text(reg_name),
            normalize_text(article_number),
            normalize_text(action),
            normalize_text(result),
        ]
    )
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:12]
    return f"rule_{digest}"


def build_fallback_rules(
    article_number: str, content: str, rule_type: str
) -> List[Dict[str, str]]:
    """
    Deterministic fallback: splits content into sentences and emits one rule
    per sentence. Type is already resolved — no keyword classification needed.
    Consecutive consequence sentences are attached as the result of the prior action.
    """
    CONSEQUENCE_SIGNALS = [
        "shall be",
        "will be",
        "must be",
        "is subject to",
        "shall result",
        "penalty",
        "deducted",
        "expelled",
        "revoked",
        "forced to",
        "zero grade",
        "sanction",
    ]

    rules = []
    sentences = re.split(r"(?<=[\.\;\:])\s+", content)

    pending_action = ""
    pending_result = ""

    def flush():
        nonlocal pending_action, pending_result
        if pending_action.strip():
            rules.append(
                {
                    "type": rule_type,
                    "action": pending_action.strip(),
                    "result": pending_result.strip(),
                    "art_ref": article_number,
                }
            )
        pending_action, pending_result = "", ""

    for sent in sentences:
        s = sent.strip()
        if len(s) < 10:
            continue

        low = s.lower()
        is_consequence = any(sig in low for sig in CONSEQUENCE_SIGNALS)

        if is_consequence and pending_action:
            pending_result = s
            flush()
        else:
            flush()
            pending_action = s
            pending_result = ""

    flush()
    return rules


def setup_graph(driver, regulations, articles, reg_map) -> None:
    """
    Short-lived setup session:
    - clear graph
    - create Regulation nodes
    - create Article nodes
    - create article fulltext index
    """
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

        for reg_id, name, category in regulations:
            session.run(
                "MERGE (r:Regulation {id:$rid}) SET r.name=$name, r.category=$cat",
                rid=reg_id,
                name=name,
                cat=category,
            )

        for reg_id, article_number, content in articles:
            reg_name, reg_category = reg_map.get(reg_id, ("Unknown", "Unknown"))
            session.run(
                """
                MATCH (r:Regulation {id: $rid})
                CREATE (a:Article {
                    number:   $num,
                    content:  $content,
                    reg_name: $reg_name,
                    category: $reg_category
                })
                MERGE (r)-[:HAS_ARTICLE]->(a)
                """,
                rid=reg_id,
                num=article_number,
                content=content,
                reg_name=reg_name,
                reg_category=reg_category,
            )

        session.run(
            """
            CREATE FULLTEXT INDEX article_content_idx IF NOT EXISTS
            FOR (a:Article) ON EACH [a.content]
            """
        )


def write_rules_batch(driver, rules_to_write: List[Dict]) -> None:
    """
    Short-lived write session for one processed batch.
    """
    with driver.session() as session:
        for item in rules_to_write:
            session.run(
                """
                MATCH (a:Article {number: $art_ref, reg_name: $reg_name})
                MERGE (z:Rule {rule_id: $rule_id})
                SET z.type = $type,
                    z.action = $action,
                    z.result = $result,
                    z.art_ref = $art_ref,
                    z.reg_name = $reg_name
                MERGE (a)-[:CONTAINS_RULE]->(z)
                """,
                rule_id=item["rule_id"],
                type=item["type"],
                action=item["action"],
                result=item["result"],
                art_ref=item["art_ref"],
                reg_name=item["reg_name"],
            )

def embed_rules(rules: List[Dict]) -> List[Dict]:
    """
    Adds action_embedding, result_embedding, combined_embedding
    to each rule dict in-place. result_embedding is None when result is empty.
    """
    embedder = get_embedder()

    action_texts    = [r["action"] for r in rules]
    result_texts    = [r["result"] for r in rules]
    combined_texts  = [
        r["action"] + (" " + r["result"] if r["result"] else "")
        for r in rules
    ]

    action_vecs   = embedder.encode(action_texts,   convert_to_numpy=True)
    combined_vecs = embedder.encode(combined_texts, convert_to_numpy=True)

    # Only embed non-empty results in a single batch
    non_empty_idx  = [i for i, r in enumerate(rules) if r["result"]]
    result_vecs    = [None] * len(rules)
    if non_empty_idx:
        batch        = [result_texts[i] for i in non_empty_idx]
        batch_vecs   = embedder.encode(batch, convert_to_numpy=True)
        for idx, vec in zip(non_empty_idx, batch_vecs):
            result_vecs[idx] = vec

    for i, rule in enumerate(rules):
        rule["action_embedding"]   = action_vecs[i].tolist()
        rule["result_embedding"]   = result_vecs[i].tolist() if result_vecs[i] is not None else None
        rule["combined_embedding"] = combined_vecs[i].tolist()

    return rules


def write_rules_batch(driver, rules_to_write: List[Dict]) -> None:
    """
    Short-lived write session for one processed batch.
    Includes embedding vectors on Rule nodes.
    """
    # Embed before opening the DB session
    rules_to_write = embed_rules(rules_to_write)

    with driver.session() as session:
        for item in rules_to_write:
            session.run(
                """
                MATCH (a:Article {number: $art_ref, reg_name: $reg_name})
                MERGE (z:Rule {rule_id: $rule_id})
                SET z.type                = $type,
                    z.action              = $action,
                    z.result              = $result,
                    z.art_ref             = $art_ref,
                    z.reg_name            = $reg_name,
                    z.action_embedding    = $action_embedding,
                    z.result_embedding    = $result_embedding,
                    z.combined_embedding  = $combined_embedding
                MERGE (a)-[:CONTAINS_RULE]->(z)
                """,
                rule_id            = item["rule_id"],
                type               = item["type"],
                action             = item["action"],
                result             = item["result"],
                art_ref            = item["art_ref"],
                reg_name           = item["reg_name"],
                action_embedding   = item["action_embedding"],
                result_embedding   = item["result_embedding"],
                combined_embedding = item["combined_embedding"],
            )


def finalize_graph(driver) -> None:
    """
    Short-lived final session:
    - fulltext index on action + result only (type dropped)
    - vector indexes on the three embedding properties (Neo4j 5.x)
    - coverage audit
    """
    with driver.session() as session:
        # Fulltext: action + result only, type removed
        session.run(
            """
            CREATE FULLTEXT INDEX rule_idx IF NOT EXISTS
            FOR (z:Rule) ON EACH [z.action, z.result]
            """
        )

        # Vector indexes — requires Neo4j 5.x Enterprise or AuraDB
        # all-MiniLM-L6-v2 produces 384-dim vectors
        for prop in ("action_embedding", "result_embedding", "combined_embedding"):
            session.run(
                f"""
                CREATE VECTOR INDEX rule_{prop}_idx IF NOT EXISTS
                FOR (z:Rule) ON z.{prop}
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: 384,
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """
            )

        coverage = session.run(
            """
            MATCH (a:Article)
            OPTIONAL MATCH (a)-[:CONTAINS_RULE]->(z:Rule)
            WITH a, count(z) AS rule_count
            RETURN count(a) AS total_articles,
                   sum(CASE WHEN rule_count > 0 THEN 1 ELSE 0 END) AS covered_articles,
                   sum(CASE WHEN rule_count = 0 THEN 1 ELSE 0 END) AS uncovered_articles
            """
        ).single()

        total_articles    = int((coverage or {}).get("total_articles",    0) or 0)
        covered_articles  = int((coverage or {}).get("covered_articles",  0) or 0)
        uncovered_articles = int((coverage or {}).get("uncovered_articles", 0) or 0)

        print(
            f"[Coverage] covered={covered_articles}/{total_articles}, "
            f"uncovered={uncovered_articles}"
        )
def build_graph() -> None:
    """Build KG from SQLite into Neo4j using the fixed assignment schema."""
    VALID_TYPES = {
        "obligation",
        "prohibition",
        "permission",
        "penalty",
        "procedure",
        "other",
    }

    sql_conn = sqlite3.connect("ncu_regulations.db")
    cursor = sql_conn.cursor()
    driver = GraphDatabase.driver(URI, auth=AUTH)

    # Warm up local LLM once
    load_local_llm()

    # Load source data from SQLite
    cursor.execute("SELECT reg_id, name, category FROM regulations")
    regulations = cursor.fetchall()
    reg_map: dict[int, tuple[str, str]] = {
        reg_id: (name, category) for reg_id, name, category in regulations
    }

    # Print resolved fallback types at startup so you can verify
    print("[Fallback type mapping from reg_name]")
    for reg_id, (name, _) in reg_map.items():
        print(f"  reg_id={reg_id} | {name!r} → {resolve_type(name)}")
    print()

    cursor.execute("SELECT reg_id, article_number, content FROM articles")
    articles = cursor.fetchall()

    # Setup graph in a short session
    setup_graph(driver, regulations, articles, reg_map)

    total_articles = len(articles)
    processed_articles = 0

    # Process in batches of 3 for batched inference
    for batch_idx, batch in enumerate(chunked(articles, BATCH_SIZE), start=1):
        batch_for_llm: List[Tuple[str, str, str]] = []
        batch_meta: List[Tuple[int, str, str, str]] = []

        for reg_id, article_number, content in batch:
            reg_name, reg_category = reg_map.get(reg_id, ("Unknown", "Unknown"))
            batch_for_llm.append((article_number, reg_name, content))
            batch_meta.append((reg_id, article_number, content, reg_name))

        batch_results = extract_entities_batch(batch_for_llm)

        rules_to_write: List[Dict] = []

        for (_, article_number, content, reg_name), result in zip(
            batch_meta, batch_results
        ):
            fallback_type = resolve_type(reg_name)
            rules = result.get("rules", [])

            if not rules:
                rules = build_fallback_rules(article_number, content, fallback_type)

            clean_rules = []
            for r in rules:
                action = r.get("action", "").strip()
                result_text = r.get("result", "").strip()
                # Use LLM's type if valid, otherwise fall back to reg_name-derived type
                rule_type = r.get("type", "").strip()
                if rule_type not in VALID_TYPES:
                    rule_type = fallback_type

                if not action:
                    continue

                clean_rules.append(
                    {
                        "type": rule_type,
                        "action": action,
                        "result": result_text,
                        "art_ref": article_number,
                        "reg_name": reg_name,
                    }
                )

            clean_rules = deduplicate_rules(clean_rules)

            # progress print
            processed_articles += 1
            print(
                f"[Progress] processed {processed_articles}/{total_articles} articles"
                f" | Article {article_number}"
            )

            if clean_rules:
                print("  Rules:")
                for idx, r in enumerate(clean_rules, start=1):
                    action_preview = r["action"][:120].replace("\n", " ")
                    result_preview = r["result"][:120].replace("\n", " ")
                    print(
                        f"    {idx}. type={r['type']} | action={action_preview}"
                        + (f" | result={result_preview}" if result_preview else "")
                    )
            else:
                print("  Rules: []")

            for r in clean_rules:
                rule_id = make_rule_id(
                    article_number,
                    reg_name,
                    r["action"],
                    r["result"],
                )

                rules_to_write.append(
                    {
                        "rule_id": rule_id,
                        "type": r["type"],
                        "action": r["action"],
                        "result": r["result"],
                        "art_ref": article_number,
                        "reg_name": reg_name,
                    }
                )

        # short-lived session per processed batch
        write_rules_batch(driver, rules_to_write)
        print(
            f"[Batch] finished batch {batch_idx} with {len(batch)} articles and {len(rules_to_write)} rules\n"
        )

    finalize_graph(driver)

    driver.close()
    sql_conn.close()


if __name__ == "__main__":
    build_graph()
