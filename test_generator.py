import json
import re
import os
from typing import Dict, Any
from dotenv import load_dotenv
from neo4j import GraphDatabase

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

# $$$$$$$$ NEW PART $$$$$$$$

if driver is None:
    print("\n❌ Neo4j driver is NOT connected. Retrieval will return empty results.\n")
else:
    print("\n✅ Neo4j driver connected successfully.\n")

    # quick sanity check query
    try:
        with driver.session() as session:
            test = session.run("RETURN 1 AS ok")
            print("DB test:", [r["ok"] for r in test])
    except Exception as e:
        print(f"❌ DB test query failed: {e}")

# $$$$$$$$ NEW PART $$$$$$$$


ALLOWED_TYPES = {
    "obligation",
    "prohibition",
    "permission",
    "penalty",
    "procedure",
    "other",
}


def count_words(text: str) -> int:
    if not text or not text.strip():
        return 0
    return len(re.findall(r"\b[\w'-]+\b", text.strip()))


def ensure_llm_loaded() -> None:
    """
    Make sure tokenizer and pipeline are initialized.
    """
    if get_tokenizer() is None or get_raw_pipeline() is None:
        load_local_llm()


def build_prompt(user_input: str, profile: str, schema_description: str) -> str:
    ensure_llm_loaded()

    tokenizer = get_tokenizer()
    messages = [
        {
            "role": "system",
            "content": f"""
You are a strict JSON generator.

Profile:
{profile}

Schema:
{schema_description}

Task:
Read the user input and return exactly one JSON object with these fields:
- type
- action
- result

Rules:
- type must be exactly one of:
  obligation, prohibition, permission, penalty, procedure, other
- action must be less than 7 words
- result must be less than 7 words, or "" if no clear result is stated
- return JSON only
- do not include markdown
- do not include explanations
- do not include extra keys
- Do not use short words like mins, use full word like minutes.
- Every JSON string value must be wrapped in double quotes
- Output must be valid JSON parsable by Python json.loads


Example input and output:

Example 1
User input:
What is the penalty for being late to class?
System Output:
{{
  "type": "penalty",
  "action": "be late to class",
  "result": "receive a penalty"
}}

Example 2
User input:
How can I renew my ID card?
System Output:
{{
  "type": "procedure",
  "action": "steps to renew ID",
  "result": "Get new ID card"
}}


Example 3
User input:
What happens if I dont pay my tuition?
System Output:
{{
  "type": "penalty",
  "action": "not paying tuition",
  "result": "consecuencess".
}}


User input:
{user_input}
""".strip(),
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def read_pipeline_text_output(raw_response: Any) -> str:
    """
    Extract generated text from a Hugging Face pipeline response.
    """
    if isinstance(raw_response, list):
        if not raw_response:
            return ""
        first = raw_response[0]
        if isinstance(first, dict):
            return first.get("generated_text", "")
        return str(first)

    if isinstance(raw_response, dict):
        return raw_response.get("generated_text", "")

    return str(raw_response)


def call_llm_once(prompt: str) -> str:
    """
    Send one prompt to the LLM and return raw text.
    """
    ensure_llm_loaded()

    llm = get_raw_pipeline()
    raw_response = llm(prompt)
    return read_pipeline_text_output(raw_response)


def parse_json_text(response_text: str) -> Dict[str, Any]:
    """
    Parse the LLM's JSON text into a Python dict.
    """
    response_text = response_text.strip()
    return json.loads(response_text)


def validate_output(data: Dict[str, Any]) -> None:
    """
    Raise ValueError if the parsed output does not match the schema.
    """
    required_keys = {"type", "action", "result"}
    actual_keys = set(data.keys())

    print("\n--- VALIDATION START ---")  # $$$$$$$$ NEW PART $$$$$$$$

    missing = required_keys - actual_keys
    extra = actual_keys - required_keys

    print(f"[CHECK] keys present: {actual_keys}")  # $$$$$$$$ NEW PART $$$$$$$$

    if missing:
        print(f"[FAIL] missing keys: {missing}")  # $$$$$$$$ NEW PART $$$$$$$$
        raise ValueError(f"Missing keys: {sorted(missing)}")
    else:
        print("[PASS] required keys present")  # $$$$$$$$ NEW PART $$$$$$$$

    if extra:
        print(f"[FAIL] unexpected keys: {extra}")  # $$$$$$$$ NEW PART $$$$$$$$
        raise ValueError(f"Unexpected keys: {sorted(extra)}")
    else:
        print("[PASS] no extra keys")  # $$$$$$$$ NEW PART $$$$$$$$

    if not isinstance(data["type"], str):
        print(
            f"[FAIL] type is not string: {data['type']}"
        )  # $$$$$$$$ NEW PART $$$$$$$$
        raise ValueError("type must be a string")

    type_value = data["type"].strip().lower()
    print(f"[CHECK] type value: '{type_value}'")  # $$$$$$$$ NEW PART $$$$$$$$

    if type_value not in ALLOWED_TYPES:
        print(f"[FAIL] invalid type: '{type_value}'")  # $$$$$$$$ NEW PART $$$$$$$$
        raise ValueError(f"type must be one of: {sorted(ALLOWED_TYPES)}")
    else:
        print("[PASS] type is valid")  # $$$$$$$$ NEW PART $$$$$$$$

    if not isinstance(data["action"], str):
        print(
            f"[FAIL] action not string: {data['action']}"
        )  # $$$$$$$$ NEW PART $$$$$$$$
        raise ValueError("action must be a string")

    action_wc = count_words(data["action"])
    print(
        f"[CHECK] action: '{data['action']}' | words={action_wc}"
    )  # $$$$$$$$ NEW PART $$$$$$$$

    if action_wc > 6:
        print("[FAIL] action too long")  # $$$$$$$$ NEW PART $$$$$$$$
        raise ValueError("action must contain fewer than 7 words")
    else:
        print("[PASS] action word count valid")  # $$$$$$$$ NEW PART $$$$$$$$

    if not isinstance(data["result"], str):
        print(
            f"[FAIL] result not string: {data['result']}"
        )  # $$$$$$$$ NEW PART $$$$$$$$
        raise ValueError("result must be a string")

    result_wc = count_words(data["result"])
    print(
        f"[CHECK] result: '{data['result']}' | words={result_wc}"
    )  # $$$$$$$$ NEW PART $$$$$$$$

    if data["result"] != "" and result_wc > 6:
        print("[FAIL] result too long")  # $$$$$$$$ NEW PART $$$$$$$$
        raise ValueError('result must contain fewer than 7 words or be ""')
    else:
        print("[PASS] result word count valid")  # $$$$$$$$ NEW PART $$$$$$$$

    print("--- VALIDATION PASSED ---\n")  # $$$$$$$$ NEW PART $$$$$$$$


def classify_user_input(
    user_input: str,
    profile: str,
    schema_description: str,
) -> Dict[str, str]:
    """
    Flow:
    1. Build prompt
    2. Send prompt to LLM
    3. Get JSON text back
    4. Parse JSON text into Python dict
    5. Validate fields
    6. Return final dict
    """
    prompt = build_prompt(
        user_input=user_input,
        profile=profile,
        schema_description=schema_description,
    )

    print("\n--- PROMPT SENT TO LLM ---")
    print(prompt)
    print("--------------------------\n")

    response_text = call_llm_once(prompt)

    print("\n--- RAW LLM OUTPUT ---")
    print(response_text)
    print("----------------------\n")

    data = parse_json_text(response_text)
    data = validate_or_repair(data)

    return data


from typing import Any

# $$$$$$$$ NEW PART $$$$$$$$


def build_typed_cypher(entities: dict[str, Any]) -> tuple[str, str]:
    """
    Build two Cypher queries from already-validated entities JSON.

    Branching logic:
    - if action is empty and result is empty: use type only
    - if action is empty: use type + result tokens
    - if result is empty: use type + action tokens
    - else: use type + action tokens + result tokens
    """

    action = str(entities.get("action", "") or "").strip().lower()
    result = str(entities.get("result", "") or "").strip().lower()

    use_action = action != ""
    use_result = result != ""

    if use_action and use_result:
        cypher_typed = """
MATCH p = (z:Rule)
WHERE z.type = $type
  AND ANY(word IN $action_tokens WHERE toLower(coalesce(z.action, "")) CONTAINS word)
  AND ANY(word IN $result_tokens WHERE toLower(coalesce(z.result, "")) CONTAINS word)
RETURN p,
       z.rule_id AS rule_id,
       z.type AS type,
       z.action AS action,
       z.result AS result,
       z.art_ref AS art_ref,
       z.reg_name AS reg_name,
       3 AS score
ORDER BY score DESC, rule_id
LIMIT 10
"""
        cypher_broad = """
MATCH p = (z:Rule)
WHERE z.type = $type
  AND (
        ANY(word IN $action_tokens WHERE toLower(coalesce(z.action, "")) CONTAINS word)
        OR
        ANY(word IN $result_tokens WHERE toLower(coalesce(z.result, "")) CONTAINS word)
      )
RETURN p,
       z.rule_id AS rule_id,
       z.type AS type,
       z.action AS action,
       z.result AS result,
       z.art_ref AS art_ref,
       z.reg_name AS reg_name,
       CASE
         WHEN ANY(word IN $action_tokens WHERE toLower(coalesce(z.action, "")) CONTAINS word)
          AND ANY(word IN $result_tokens WHERE toLower(coalesce(z.result, "")) CONTAINS word) THEN 3
         WHEN ANY(word IN $action_tokens WHERE toLower(coalesce(z.action, "")) CONTAINS word)
           OR ANY(word IN $result_tokens WHERE toLower(coalesce(z.result, "")) CONTAINS word) THEN 2
         ELSE 1
       END AS score
ORDER BY score DESC, rule_id
LIMIT 15
"""

    elif use_action:
        cypher_typed = """
MATCH p = (z:Rule)
WHERE z.type = $type
  AND ANY(word IN $action_tokens WHERE toLower(coalesce(z.action, "")) CONTAINS word)
RETURN p,
       z.rule_id AS rule_id,
       z.type AS type,
       z.action AS action,
       z.result AS result,
       z.art_ref AS art_ref,
       z.reg_name AS reg_name,
       2 AS score
ORDER BY score DESC, rule_id
LIMIT 10
"""
        cypher_broad = """
MATCH p = (z:Rule)
WHERE z.type = $type
   OR ANY(word IN $action_tokens WHERE toLower(coalesce(z.action, "")) CONTAINS word)
RETURN p,
       z.rule_id AS rule_id,
       z.type AS type,
       z.action AS action,
       z.result AS result,
       z.art_ref AS art_ref,
       z.reg_name AS reg_name,
       CASE
         WHEN z.type = $type
          AND ANY(word IN $action_tokens WHERE toLower(coalesce(z.action, "")) CONTAINS word) THEN 2
         WHEN z.type = $type
           OR ANY(word IN $action_tokens WHERE toLower(coalesce(z.action, "")) CONTAINS word) THEN 1
         ELSE 0
       END AS score
ORDER BY score DESC, rule_id
LIMIT 15
"""

    elif use_result:
        cypher_typed = """
MATCH p = (z:Rule)
WHERE z.type = $type
  AND ANY(word IN $result_tokens WHERE toLower(coalesce(z.result, "")) CONTAINS word)
RETURN p,
       z.rule_id AS rule_id,
       z.type AS type,
       z.action AS action,
       z.result AS result,
       z.art_ref AS art_ref,
       z.reg_name AS reg_name,
       2 AS score
ORDER BY score DESC, rule_id
LIMIT 10
"""
        cypher_broad = """
MATCH p = (z:Rule)
WHERE z.type = $type
   OR ANY(word IN $result_tokens WHERE toLower(coalesce(z.result, "")) CONTAINS word)
RETURN p,
       z.rule_id AS rule_id,
       z.type AS type,
       z.action AS action,
       z.result AS result,
       z.art_ref AS art_ref,
       z.reg_name AS reg_name,
       CASE
         WHEN z.type = $type
          AND ANY(word IN $result_tokens WHERE toLower(coalesce(z.result, "")) CONTAINS word) THEN 2
         WHEN z.type = $type
           OR ANY(word IN $result_tokens WHERE toLower(coalesce(z.result, "")) CONTAINS word) THEN 1
         ELSE 0
       END AS score
ORDER BY score DESC, rule_id
LIMIT 15
"""

    else:
        cypher_typed = """
MATCH p = (z:Rule)
WHERE z.type = $type
RETURN p,
       z.rule_id AS rule_id,
       z.type AS type,
       z.action AS action,
       z.result AS result,
       z.art_ref AS art_ref,
       z.reg_name AS reg_name,
       1 AS score
ORDER BY score DESC, rule_id
LIMIT 10
"""
        cypher_broad = """
MATCH p = (z:Rule)
WHERE z.type = $type
RETURN p,
       z.rule_id AS rule_id,
       z.type AS type,
       z.action AS action,
       z.result AS result,
       z.art_ref AS art_ref,
       z.reg_name AS reg_name,
       1 AS score
ORDER BY score DESC, rule_id
LIMIT 15
"""

    return cypher_typed, cypher_broad


# $$$$$$$$ NEW PART $$$$$$$$

# $$$$$$$$ NEW PART $$$$$$$$

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "from",
    "than",
    "more",
    "less",
    "be",
    "is",
    "are",
    "was",
    "were",
    "x",
    "will",
    "shall",
    "should",
    "can",
    "could",
}


def tokenize_for_retrieval(text: str) -> list[str]:
    tokens = re.findall(r"\b[\w'-]+\b", text.lower())
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


# $$$$$$$$ NEW PART $$$$$$$$


def build_typed_params(entities: dict[str, Any]) -> dict[str, Any]:
    # $$$$$$$$ NEW PART $$$$$$$$
    action_tokens = tokenize_for_retrieval(entities["action"])
    result_tokens = tokenize_for_retrieval(entities["result"])

    print("\n=== TOKEN DEBUG ===")
    print("Raw action:", entities["action"])
    print("Action tokens:", action_tokens)
    print("Raw result:", entities["result"])
    print("Result tokens:", result_tokens)
    # $$$$$$$$ NEW PART $$$$$$$$

    return {
        "type": entities["type"].strip().lower(),
        # $$$$$$$$ NEW PART $$$$$$$$
        "action_tokens": action_tokens,
        "result_tokens": result_tokens,
        # $$$$$$$$ NEW PART $$$$$$$$
    }


import json
from typing import Dict, Any


def build_repair_prompt(data: Dict[str, Any]) -> str:
    ensure_llm_loaded()

    tokenizer = get_tokenizer()
    messages = [
        {
            "role": "system",
            "content": f"""
You are a strict JSON repairer.

Task:
You will receive a JSON object with fields:
- type
- action
- result

Your job is to rewrite it so it passes the schema rules below.

Schema rules:
- Output must remain a JSON object with exactly these keys:
  - type
  - action
  - result
- type must be exactly one of:
  obligation, prohibition, permission, penalty, procedure, other
- If type is invalid, replace it with the closest valid type.
- action must contain fewer than 6 words.
- result must contain fewer than 6 words, or be "".
- Keep the meaning as close as possible.
- Do not add extra keys.
- Return JSON only.
- Do not use markdown.
- Do not explain.

Input JSON:
{json.dumps(data, ensure_ascii=False, indent=2)}
""".strip(),
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def repair_with_llm(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send invalid parsed data to the LLM and ask it to rewrite the JSON
    so it passes validation.
    """
    prompt = build_repair_prompt(data)
    response_text = call_llm_once(prompt)

    print("\n--- REPAIR PROMPT SENT TO LLM ---")
    print(prompt)
    print("---------------------------------\n")

    print("\n--- RAW REPAIRED LLM OUTPUT ---")
    print(response_text)
    print("-------------------------------\n")

    repaired_data = parse_json_text(response_text)
    return repaired_data


def validate_or_repair(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    First try normal validation.
    If ValueError happens, repair with LLM and validate again.
    """
    try:
        validate_output(data)
        return data
    except ValueError as e:
        print(f"\nValidation failed: {e}")
        print("Attempting LLM repair...\n")

    repaired_data = repair_with_llm(data)
    validate_output(repaired_data)
    return repaired_data


# $$$$$$$$ NEW PART $$$$$$$$


def _tokenize_for_overlap(text: str) -> set[str]:
    if not text:
        return set()
    return set(re.findall(r"\b[\w'-]+\b", text.lower()))


def _compute_python_rank(row: dict[str, Any], entities: dict[str, Any]) -> int:
    """
    Python-side reranking.

    Score components:
    - typed source gets a strong boost
    - DB score still matters
    - token overlap on action/result refines ranking
    """
    source = str(row.get("source", "") or "").strip().lower()
    db_score = int(row.get("score", 0) or 0)

    query_action_tokens = _tokenize_for_overlap(str(entities.get("action", "") or ""))
    query_result_tokens = _tokenize_for_overlap(str(entities.get("result", "") or ""))

    row_action_tokens = _tokenize_for_overlap(str(row.get("action", "") or ""))
    row_result_tokens = _tokenize_for_overlap(str(row.get("result", "") or ""))

    action_overlap = len(query_action_tokens & row_action_tokens)
    result_overlap = len(query_result_tokens & row_result_tokens)

    source_boost = 100 if source == "typed" else 0

    return (
        source_boost + (30 * db_score) + (10 * action_overlap) + (10 * result_overlap)
    )


def _run_query_with_source(
    query: str,
    params: dict[str, Any],
    source: str,
) -> list[dict[str, Any]]:
    """
    Run one Cypher query and tag each row with its source.
    """
    if driver is None:
        return []

    with driver.session() as session:
        records = session.run(query, params)
        rows = [dict(record) for record in records]

    for row in rows:
        row["source"] = source

    return rows


def _merge_ranked_results(
    typed_rows: list[dict[str, Any]],
    broad_rows: list[dict[str, Any]],
    entities: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Merge duplicates by rule_id and keep the higher-ranked version.
    """
    merged: dict[str, dict[str, Any]] = {}

    for row in typed_rows + broad_rows:
        rule_id = str(row.get("rule_id", "") or "").strip()
        if not rule_id:
            continue

        row["python_rank"] = _compute_python_rank(row, entities)

        if rule_id not in merged:
            merged[rule_id] = row
            continue

        if row["python_rank"] > merged[rule_id]["python_rank"]:
            merged[rule_id] = row

    return sorted(
        merged.values(),
        key=lambda x: (
            x.get("python_rank", 0),
            x.get("score", 0),
            x.get("rule_id", ""),
        ),
        reverse=True,
    )


def get_relevant_articles(
    entities: dict[str, Any],
    typed_query: str,
    broad_query: str,
    params: dict[str, Any],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Run typed+broad retrieval, tag source, rank in Python, merge duplicates,
    and return top-k rows.
    """
    if driver is None:
        return []

    print("\n=== RETRIEVAL INPUT ENTITIES ===")
    print(json.dumps(entities, indent=2, ensure_ascii=False))

    print("\n=== RETRIEVAL PARAMS ===")
    print(json.dumps(params, indent=2, ensure_ascii=False))

    print("\n=== RUNNING TYPED QUERY ===")
    typed_rows = _run_query_with_source(typed_query, params, source="typed")
    print(f"Typed hits: {len(typed_rows)}")

    print("\n=== RUNNING BROAD QUERY ===")
    broad_rows = _run_query_with_source(broad_query, params, source="broad")
    print(f"Broad hits: {len(broad_rows)}")

    ranked_rows = _merge_ranked_results(typed_rows, broad_rows, entities)

    print("\n=== FINAL RANKED RESULTS ===")
    for idx, row in enumerate(ranked_rows[:top_k], start=1):
        print(
            f"[{idx}] "
            f"rule_id={row.get('rule_id')} | "
            f"source={row.get('source')} | "
            f"db_score={row.get('score')} | "
            f"python_rank={row.get('python_rank')} | "
            f"type={row.get('type')} | "
            f"art_ref={row.get('art_ref')}"
        )

    return ranked_rows[:top_k]


# $$$$$$$$ NEW PART $$$$$$$$


# $$$$$$$$ NEW PART $$$$$$$$


def _format_rules_for_generation(
    rule_results: list[dict[str, Any]], max_rules: int = 5
) -> str:
    """
    Turn retrieved rule dicts into compact text context for the generator.
    """
    lines = []

    for i, rule in enumerate(rule_results[:max_rules], start=1):
        lines.append(
            f"""Rule {i}:
- rule_id: {rule.get("rule_id", "")}
- reg_name: {rule.get("reg_name", "")}
- art_ref: {rule.get("art_ref", "")}
- type: {rule.get("type", "")}
- action: {rule.get("action", "")}
- result: {rule.get("result", "")}
- source: {rule.get("source", "")}
- retrieval_score: {rule.get("python_rank", "")}
"""
        )

    return "\n".join(lines)


# $$$$$$$$ NEW PART $$$$$$$$


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


# $$$$$$$$ NEW PART $$$$$$$$


def generate_answer(question: str, rule_results: list[dict[str, Any]]) -> str:
    """
    Generate a final answer from retrieved KG rule results.

    Behavior:
    - If no rules are retrieved, return a fallback answer.
    - Otherwise, ask the local model to answer using only the provided rules.
    """
    if not rule_results:
        return "I don't know based on the retrieved rules."

    rules_context = _format_rules_for_generation(rule_results, max_rules=5)

    messages = [
        {
            "role": "system",
            "content": """
You are a careful regulation QA assistant.

- Answer the user's question using ONLY the retrieved rules provided.
Do not invent facts.
- If the rule gives a partial answer, explain the limitation
- Do NOT default to "I don't know" if some relevant rule exists
- Only say "I don't know" if no relevant rule exists at all

Instructions:
- Prefer the highest-ranked and most directly relevant rule.
- If multiple rules are relevant, synthesize them briefly.
- Cite the supporting source at the end in this format:
  [Source: <reg_name>, <art_ref>]
- If more than one source is used, include multiple citations.
- Be concise and direct.
""".strip(),
        },
        {
            "role": "user",
            "content": f"""
Question:
{question}

Retrieved rules:
{rules_context}

Write the final answer.
""".strip(),
        },
    ]

    answer = generate_text(messages, max_new_tokens=220)
    return answer.strip()


# $$$$$$$$ NEW PART $$$$$$$$


if __name__ == "__main__":

    question = "Can I  leave an exam early if I finish ahead of time?"
    profile = (
        "Classify the user request into exactly one legal-style category "
        "and summarize the action and result under the required schema."
    )

    schema_description = """
{
  "type": "choose exactly one of: obligation, prohibition, permission, penalty, procedure, other",
  "action": " 6 or less words",
  "result": " 6 or less words, if not posisble use empty string"
}
""".strip()

    output = classify_user_input(
        user_input=question,
        profile=profile,
        schema_description=schema_description,
    )

    print("Parsed output:")
    print(json.dumps(output, indent=2, ensure_ascii=False))

    entities = output

    typed_query, broad_query = build_typed_cypher(entities)

    print("\nTyped Cypher:")
    print(typed_query)
    print("\nBroad Cypher:")
    print(broad_query)
    params = build_typed_params(entities)

    print("\nParams:")
    print(params)

    # $$$$$$$$ NEW PART $$$$$$$$

    relevant_rules = get_relevant_articles(
        entities=entities,
        typed_query=typed_query,
        broad_query=broad_query,
        params=params,
        top_k=10,
    )

    print("\nRelevant rules:")
    print(json.dumps(relevant_rules, indent=2, ensure_ascii=False, default=str))

    # $$$$$$$$ NEW PART $$$$$$$$

    # $$$$$$$$ NEW PART $$$$$$$$

    final_answer = generate_answer(
        question=question,
        rule_results=relevant_rules,
    )

    print("\nFinal answer:")
    print(final_answer)

# $$$$$$$$ NEW PART $$$$$$$$
