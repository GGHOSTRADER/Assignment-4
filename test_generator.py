import json
import re
from typing import Dict, Any

from llm_loader import load_local_llm, get_tokenizer, get_raw_pipeline

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
- action must be exactly 4 words
- result must be exactly 4 words, or "" if no clear result is stated
- return JSON only
- do not include markdown
- do not include explanations
- do not include extra keys

Example output:
{{
  "type": "procedure",
  "action": "explain passport renewal steps",
  "result": "complete passport renewal process"
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

    missing = required_keys - actual_keys
    extra = actual_keys - required_keys

    if missing:
        raise ValueError(f"Missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected keys: {sorted(extra)}")

    if not isinstance(data["type"], str):
        raise ValueError("type must be a string")
    if data["type"] not in ALLOWED_TYPES:
        raise ValueError(f"type must be one of: {sorted(ALLOWED_TYPES)}")

    if not isinstance(data["action"], str):
        raise ValueError("action must be a string")
    if count_words(data["action"]) > 6:
        raise ValueError("action must be less than 7 words ")

    if not isinstance(data["result"], str):
        raise ValueError("result must be a string")
    if data["result"] != "" and count_words(data["result"]) > 6:
        raise ValueError('result less than 7 words or be ""')


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
    validate_output(data)

    return data


from typing import Any


def build_typed_cypher(entities: dict[str, Any]) -> tuple[str, str]:
    """
    Build two Cypher queries from already-validated entities JSON.

    Expected input:
    {
        "type": "obligation" | "prohibition" | "permission" | "penalty" | "procedure" | "other",
        "action": "...",
        "result": "..."
    }

    Branching logic:
    - if action is empty and result is empty: use type only
    - if action is empty: use type + result
    - if result is empty: use type + action
    - else: use type + action + result
    """

    rule_type = str(entities.get("type", "") or "").strip().lower()
    action = str(entities.get("action", "") or "").strip().lower()
    result = str(entities.get("result", "") or "").strip().lower()

    use_action = action != ""
    use_result = result != ""

    if use_action and use_result:
        cypher_typed = """
MATCH p = (z:Rule)
WHERE z.type = $type
  AND toLower(coalesce(z.action, "")) CONTAINS $action
  AND toLower(coalesce(z.result, "")) CONTAINS $result
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
        toLower(coalesce(z.action, "")) CONTAINS $action
        OR toLower(coalesce(z.result, "")) CONTAINS $result
      )
RETURN p,
       z.rule_id AS rule_id,
       z.type AS type,
       z.action AS action,
       z.result AS result,
       z.art_ref AS art_ref,
       z.reg_name AS reg_name,
       CASE
         WHEN toLower(coalesce(z.action, "")) CONTAINS $action
              AND toLower(coalesce(z.result, "")) CONTAINS $result THEN 3
         WHEN toLower(coalesce(z.action, "")) CONTAINS $action
              OR toLower(coalesce(z.result, "")) CONTAINS $result THEN 2
         ELSE 1
       END AS score
ORDER BY score DESC, rule_id
LIMIT 15
"""

    elif use_action:
        cypher_typed = """
MATCH p = (z:Rule)
WHERE z.type = $type
  AND toLower(coalesce(z.action, "")) CONTAINS $action
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
   OR toLower(coalesce(z.action, "")) CONTAINS $action
RETURN p,
       z.rule_id AS rule_id,
       z.type AS type,
       z.action AS action,
       z.result AS result,
       z.art_ref AS art_ref,
       z.reg_name AS reg_name,
       CASE
         WHEN z.type = $type
              AND toLower(coalesce(z.action, "")) CONTAINS $action THEN 2
         WHEN z.type = $type
              OR toLower(coalesce(z.action, "")) CONTAINS $action THEN 1
         ELSE 0
       END AS score
ORDER BY score DESC, rule_id
LIMIT 15
"""

    elif use_result:
        cypher_typed = """
MATCH p = (z:Rule)
WHERE z.type = $type
  AND toLower(coalesce(z.result, "")) CONTAINS $result
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
   OR toLower(coalesce(z.result, "")) CONTAINS $result
RETURN p,
       z.rule_id AS rule_id,
       z.type AS type,
       z.action AS action,
       z.result AS result,
       z.art_ref AS art_ref,
       z.reg_name AS reg_name,
       CASE
         WHEN z.type = $type
              AND toLower(coalesce(z.result, "")) CONTAINS $result THEN 2
         WHEN z.type = $type
              OR toLower(coalesce(z.result, "")) CONTAINS $result THEN 1
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


def build_typed_params(entities: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": entities["type"].strip().lower(),
        "action": entities["action"].strip().lower(),
        "result": entities["result"].strip().lower(),
    }


if __name__ == "__main__":
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
        user_input="How many minutes late can a student be before they are barred from the exam?",
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
