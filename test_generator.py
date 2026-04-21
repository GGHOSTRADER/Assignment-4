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

Example input and output:

Example 1
User input:
What is the penalty for being late to class?
System Output:
{{
  "type": "penalty",
  "action": be late to class,
  "result": receive a penalty
}}

Example 2
User input:
How can I renew my ID card?
System Output:
{{
  "type": "procedure",
  "action": steps to renew ID,
  "result": Get new ID card
}}


Example 3
User input:
What happens if I dont pay my tuition?
System Output:
{{
  "type": "penalty",
  "action": not paying tuition,
  "result": consecuencess.
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
        user_input="How many minutes late can a student be before they are barred from the exam? since I will be running late this morning and I dont wanna miss more classes.",
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
