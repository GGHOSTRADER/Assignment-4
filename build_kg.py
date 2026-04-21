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
from typing import List, Dict, Tuple, Iterable
from dotenv import load_dotenv
from neo4j import GraphDatabase
import hashlib
import json
import re

from llm_loader import load_local_llm, get_tokenizer, get_raw_pipeline


# ========== 0) Initialization ==========
load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (
    os.getenv("NEO4J_USER", "neo4j"),
    os.getenv("NEO4J_PASSWORD", "password"),
)

BATCH_SIZE = 3


def chunked(seq: List[Tuple], size: int) -> Iterable[List[Tuple]]:
    """Yield successive chunks from a list."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def build_prompt(article_number: str, reg_name: str, content: str) -> str:
    tokenizer = get_tokenizer()

    messages = [
        {
            "role": "system",
            "content": f"""
Extract rules from the regulation text below.

Field definitions:
- type: choose exactly one of:
  obligation, prohibition, permission, penalty, procedure, other
- action: the main behavior, requirement, restriction, or allowed act described in the text
- result: the consequence, outcome, punishment, or follow-up stated in the text; use "" if none is stated

Rules:
- Extract one or more rules if present.
- Keep wording close to the original text.
- Do not invent facts.
- If the text contains no clear rule, return {{"rules": []}}.

Example:
Text: Students must bring their student ID to the exam. Violators shall have five points deducted from their exam grade.
Output:
{{
  "rules": [
    {{
      "type": "obligation",
      "action": "Students must bring their student ID to the exam",
      "result": "Violators shall have five points deducted from their exam grade"
    }}
  ]
}}

Example:
Text: Students go to class.
Output:
{{
  "rules": []
}}

Return JSON only in this format:
{{
  "rules": [
    {{
      "type": "...",
      "action": "...",
      "result": "..."
    }}
  ]
}}

Regulation name: {reg_name}
Article reference: {article_number}

Text:
{content}
""",
        },
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False)


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
    """
    Remove exact logical duplicates based on normalized action + result.
    Keeps the first occurrence.
    """
    seen = set()
    unique_rules = []

    for rule in rules:
        action = normalize_text(rule.get("action", ""))
        result = normalize_text(rule.get("result", ""))
        key = (action, result)

        if not action:
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


def build_fallback_rules(article_number: str, content: str) -> List[Dict[str, str]]:
    """
    Deterministic fallback: extract rules using pattern matching (no LLM).

    Improvements:
    - supports all target types:
      obligation, prohibition, permission, penalty, procedure, other
    - avoids overwriting a pending action before saving it
    - flushes final pending action at the end
    - allows penalty sentences to attach to the previous action
    """
    rules: List[Dict[str, str]] = []

    sentences = re.split(r"(?<=[\.\;\:])\s+", content)

    pending_action = ""
    pending_type = ""

    def add_rule(rule_type: str, action: str, result: str = "") -> None:
        action = action.strip()
        result = result.strip()
        if not action:
            return
        rules.append(
            {
                "type": rule_type,
                "action": action,
                "result": result,
                "art_ref": article_number,
            }
        )

    def flush_pending() -> None:
        nonlocal pending_action, pending_type
        if pending_action:
            add_rule(pending_type or "other", pending_action, "")
            pending_action = ""
            pending_type = ""

    def classify_sentence(s: str) -> str:
        if any(
            x in s
            for x in [
                "violators shall",
                "shall be punished",
                "shall receive",
                "will receive",
                "subject to disciplinary action",
                "shall be subject to",
                "will be subject to",
                "penalty",
                "punishment",
                "deducted from",
            ]
        ):
            return "penalty"

        if any(
            x in s
            for x in [
                "must",
                "shall",
                "is required to",
                "are required to",
                "has to",
                "have to",
            ]
        ):
            return "obligation"

        if any(
            x in s
            for x in [
                "not permitted",
                "may not",
                "shall not",
                "must not",
                "are not permitted",
                "is prohibited",
                "are prohibited",
                "cannot",
            ]
        ):
            return "prohibition"

        if any(
            x in s
            for x in [
                "may",
                "is allowed to",
                "are allowed to",
                "is permitted to",
                "are permitted to",
            ]
        ):
            return "permission"

        if any(
            x in s
            for x in [
                "submit",
                "apply",
                "register",
                "complete",
                "fill out",
                "provide",
                "attach",
                "follow the procedure",
                "according to the procedure",
                "within",
                "before",
                "after",
                "by the deadline",
            ]
        ):
            return "procedure"

        return "other"

    for sent in sentences:
        sent = sent.strip()
        s = sent.lower()

        if not s or len(s) < 10:
            continue

        sent_type = classify_sentence(s)

        if sent_type == "penalty":
            if pending_action:
                add_rule("penalty", pending_action, sent)
                pending_action = ""
                pending_type = ""
            else:
                add_rule("penalty", sent, "")
            continue

        if sent_type in {
            "obligation",
            "prohibition",
            "permission",
            "procedure",
            "other",
        }:
            flush_pending()
            pending_action = sent
            pending_type = sent_type
            continue

    flush_pending()
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


def finalize_graph(driver) -> None:
    """
    Short-lived final session:
    - create rule fulltext index
    - coverage audit
    """
    with driver.session() as session:
        session.run(
            """
            CREATE FULLTEXT INDEX rule_idx IF NOT EXISTS
            FOR (z:Rule) ON EACH [z.action, z.result]
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

        total_articles = int((coverage or {}).get("total_articles", 0) or 0)
        covered_articles = int((coverage or {}).get("covered_articles", 0) or 0)
        uncovered_articles = int((coverage or {}).get("uncovered_articles", 0) or 0)

        print(
            f"[Coverage] covered={covered_articles}/{total_articles}, "
            f"uncovered={uncovered_articles}"
        )


def build_graph() -> None:
    """Build KG from SQLite into Neo4j using the fixed assignment schema."""
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
            rules = result.get("rules", [])

            if not rules:
                rules = build_fallback_rules(article_number, content)

            clean_rules = []
            for r in rules:
                action = r.get("action", "").strip()
                result_text = r.get("result", "").strip()

                if not action:
                    continue

                clean_rules.append(
                    {
                        "type": r.get("type", "unknown"),
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
                f"[Progress] processed {processed_articles}/{total_articles} articles | Article {article_number}"
            )

            # quick rule preview
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
