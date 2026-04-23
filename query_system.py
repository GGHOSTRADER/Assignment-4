import json
import re
import os
from typing import Dict, Any
from dotenv import load_dotenv
from neo4j import GraphDatabase

# nltk for word variations
import nltk
from nltk.corpus import wordnet

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

if driver is None:
    print("\n❌ Neo4j driver is NOT connected. Retrieval will return empty results.\n")
else:
    print("\n✅ Neo4j driver connected successfully.\n")
    try:
        with driver.session() as session:
            test = session.run("RETURN 1 AS ok")
            print("DB test:", [r["ok"] for r in test])
    except Exception as e:
        print(f"❌ DB test query failed: {e}")

# Download wordnet data if not already present
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    print("[nltk] Downloading wordnet...")
    nltk.download("wordnet")
    nltk.download("omw-1.4")


# ========== 1) Constants ==========

ALLOWED_TYPES = {
    "obligation",
    "prohibition",
    "permission",
    "penalty",
    "procedure",
    "other",
}

# Words removed before tokenization — functional/structural words with no search value
STOPWORDS = {
    # articles and conjunctions
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
    "will",
    "shall",
    "should",
    "can",
    "could",
    # pronouns
    "my",
    "me",
    "i",
    "we",
    "us",
    "you",
    "he",
    "she",
    "they",
    "them",
    "their",
    "his",
    "her",
    "its",
    # question words
    "do",
    "how",
    "what",
    "when",
    "where",
    "why",
    "who",
    "which",
    # prepositions
    "if",
    "for",
    "in",
    "on",
    "at",
    "by",
    "with",
    "about",
    "this",
    "that",
    "it",
    "not",
    "no",
    "yes",
    # common verbs with no search value
    "get",
    "got",
    "have",
    "has",
    "had",
    "been",
    # quantifiers and determiners
    "any",
    "all",
    "some",
    "would",
    "may",
    "might",
    "also",
    "many",
    "much",
    "few",
    "each",
    "such",
    "per",
    # time/order words too generic to search
    "before",
    "after",
    "during",
    "while",
    "until",
    "since",
    "these",
    "those",
    "then",
    "than",
    "just",
    "only",
    # single letters and noise
    "x",
}

# Generic domain words — appear in almost every rule, expand last to save token slots
LOW_PRIORITY_EXPAND = {
    "student",
    "students",
    "exam",
    "exams",
    "course",
    "courses",
    "university",
    "ncu",
    "school",
    "semester",
    "academic",
    "study",
    "studies",
    "rule",
    "rules",
    "regulation",
    "regulations",
    "article",
    "department",
    "program",
    "degree",
}

# Max total tokens after expansion (configurable)
MAX_EXPANDED_TOKENS = 50

# Weight multiplier for base tokens vs expanded tokens
BASE_TOKEN_WEIGHT = 3


# ========== 2) Helpers ==========


def count_words(text: str) -> int:
    if not text or not text.strip():
        return 0
    return len(re.findall(r"\b[\w'-]+\b", text.strip()))


def tokenize_for_retrieval(text: str) -> list[str]:
    """
    Tokenize raw text — split into words, lowercase,
    remove stopwords and very short tokens.
    No summarization — all meaningful words are kept.
    """
    tokens = re.findall(r"\b[\w'-]+\b", text.lower())
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


def ensure_llm_loaded() -> None:
    if get_tokenizer() is None or get_raw_pipeline() is None:
        load_local_llm()


# ========== 3) Type Classification ==========


def build_type_classification_prompt(user_input: str) -> str:
    ensure_llm_loaded()
    tokenizer = get_tokenizer()
    messages = [
        {
            "role": "system",
            "content": f"""
You are a strict classifier.

Task:
Read the user input and return exactly one JSON object with ONE field only:
- type

Rules:
- type must be exactly one of: obligation, prohibition, permission, penalty, procedure, other
- return JSON only
- do not include markdown
- do not include explanations
- do not include any other keys

Type definitions:
- obligation: the question asks about something that MUST or SHALL be done (e.g. requirements, mandatory actions, duties).
- prohibition: the question asks about something that is NOT allowed or MUST NOT be done (e.g. banned behaviors, restrictions).
- permission: the question asks whether something IS allowed or MAY be done (e.g. "can I...", "am I allowed to...").
- penalty: the question asks about consequences, punishments, or deductions for violations (e.g. "what happens if I...", "what is the penalty for...").
- procedure: the question asks about steps, deadlines, processes, approvals, or how to apply for something (e.g. "how do I...", "what is the process for...").
- other: definitions, scope, or anything that does not fit above.

Example input and output:

Example 1
User input: How many minutes late can a student be before they are barred from the exam?
Output: {{"type": "penalty"}}

Example 2
User input: How can I renew my ID card?
Output: {{"type": "procedure"}}

Example 3
User input: Can I count my military training credits towards graduation?
Output: {{"type": "permission"}}

Example 4
User input: What happens if I am caught cheating in an exam?
Output: {{"type": "penalty"}}

Example 5
User input: What must I bring to the exam?
Output: {{"type": "obligation"}}

Example 6
User input: Am I allowed to leave the exam room early?
Output: {{"type": "permission"}}

User input: {user_input}
""".strip(),
        }
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def read_pipeline_text_output(raw_response: Any) -> str:
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
    ensure_llm_loaded()
    llm = get_raw_pipeline()
    raw_response = llm(prompt)
    return read_pipeline_text_output(raw_response)


def parse_json_text(response_text: str) -> Dict[str, Any]:
    response_text = response_text.strip()
    return json.loads(response_text)


def classify_type(user_input: str) -> str:
    """
    Ask the LLM to classify the question into one of the allowed types.
    Returns the type string. Falls back to "other" if classification fails.
    """
    prompt = build_type_classification_prompt(user_input)
    response_text = call_llm_once(prompt)

    print("\n--- TYPE CLASSIFICATION OUTPUT ---")
    print(response_text)
    print("----------------------------------\n")

    try:
        data = parse_json_text(response_text)
        type_value = str(data.get("type", "")).strip().lower()

        if type_value not in ALLOWED_TYPES:
            print(
                f"[classify_type] Invalid type '{type_value}', falling back to 'other'"
            )
            return "other"

        print(f"[classify_type] Classified as: '{type_value}'")
        return type_value

    except (json.JSONDecodeError, TypeError) as e:
        print(f"[classify_type] Parse error: {e}, falling back to 'other'")
        return "other"


# ========== 4) Token Expansion ==========


def _get_wordnet_variations(token: str) -> list[str]:
    variations = set()
    for synset in wordnet.synsets(token):
        for lemma in synset.lemmas():
            word = lemma.name().replace("_", " ").lower()
            if len(word.split()) == 1 and len(word) > 2:
                variations.add(word)
    variations.discard(token.lower())
    return list(variations)


def _build_synonym_prompt(token: str) -> str:
    return f"""You are a language expansion assistant for a university regulation search engine.

For the word "{token}" in the context of university academic regulations, generate:
- synonyms
- word variations (plural, singular, verb forms, adjective forms, noun forms)
- related terms used in academic or legal documents

Rules:
- Return a flat JSON array of strings only
- All words must be lowercase
- No duplicates
- No explanations
- No markdown
- Do not include the original word "{token}" in the output

Example output format:
["word1", "word2", "word3"]

Word to expand: "{token}"
"""


def _parse_llm_expansion(response_text: str) -> list[str]:
    try:
        clean = re.sub(r"```[a-z]*", "", response_text).replace("```", "").strip()
        match = re.search(r"\[.*?\]", clean, re.DOTALL)
        if not match:
            print(f"[expansion] No JSON array found in: {response_text[:100]}")
            return []
        raw = match.group()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[expansion] JSON parse failed, attempting bare-word recovery")
            parsed = re.findall(r"[\w'-]+", raw)
        return [
            str(w).strip().lower()
            for w in parsed
            if isinstance(w, str) and len(w.strip()) > 2
        ]
    except (json.JSONDecodeError, TypeError) as e:
        print(f"[expansion] Parse error: {e} | text: {response_text[:100]}")
        return []


def _expand_token_with_llm(token: str) -> list[str]:
    ensure_llm_loaded()
    prompt = _build_synonym_prompt(token)
    tok = get_tokenizer()
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    llm = get_raw_pipeline()
    raw = llm(formatted_prompt, max_new_tokens=300)
    if isinstance(raw, list) and raw:
        response_text = raw[0].get("generated_text", "")
    else:
        response_text = str(raw)
    expanded = _parse_llm_expansion(response_text)
    print(f"  [LLM expand] '{token}' → {expanded}")
    return expanded


def expand_tokens(
    tokens: list[str],
    max_total: int = MAX_EXPANDED_TOKENS,
) -> list[str]:
    """
    Expand tokens using WordNet + LLM synonyms.

    Key behaviors:
    1. Specific/rare tokens expand first (LOW_PRIORITY_EXPAND tokens expand last)
       so cap doesn't prevent important tokens like "barred" from expanding.
    2. Original tokens always preserved in final list.
    3. Total capped at max_total.

    Example priority order for "minutes late student barred exam":
        high priority (expand first): "minutes", "late", "barred"
        low priority (expand last):   "student", "exam"
    """
    print(f"\n=== TOKEN EXPANSION (max={max_total}) ===")
    print(f"Original tokens: {tokens}")

    # Sort: specific tokens first, generic (LOW_PRIORITY) tokens last
    sorted_tokens = sorted(tokens, key=lambda t: t in LOW_PRIORITY_EXPAND)
    print(f"Expansion order: {sorted_tokens}")

    expanded_set = set(tokens)
    ordered = list(tokens)  # original tokens always preserved first

    for token in sorted_tokens:
        print(f"\n[expanding] '{token}'")
        wn_variations = _get_wordnet_variations(token)
        print(f"  [WordNet] '{token}' → {wn_variations}")
        llm_synonyms = _expand_token_with_llm(token)

        for word in wn_variations + llm_synonyms:
            word = word.strip().lower()
            if word and word not in expanded_set and word not in STOPWORDS:
                expanded_set.add(word)
                ordered.append(word)

        if len(ordered) >= max_total:
            print(f"  [cap] Reached max_total={max_total}, stopping early.")
            break

    # Always preserve originals — fill remaining slots with expansions
    original_set = set(tokens)
    expansions = [t for t in ordered if t not in original_set]
    slots = max(0, max_total - len(tokens))
    final = tokens + expansions[:slots]

    print(f"\n=== EXPANSION RESULT ===")
    print(f"Original count : {len(tokens)}")
    print(f"Expanded count : {len(final)}")
    print(f"Original tokens guaranteed: {tokens}")
    print(f"Expanded tokens: {final}")

    return final


# ========== 5) Params Builder ==========


def build_voting_params(
    user_q: str,
    question_type: str,
    max_total: int = MAX_EXPANDED_TOKENS,
) -> dict[str, Any]:
    base_tokens = tokenize_for_retrieval(user_q)

    print("\n=== TOKEN DEBUG ===")
    print("Raw question :", user_q)
    print("Base tokens  :", base_tokens)

    expanded_tokens = expand_tokens(base_tokens, max_total=max_total)

    return {
        "type": question_type,
        "base_tokens": base_tokens,
        "all_tokens": expanded_tokens,
    }


# ========== 6) Cypher Queries ==========


def build_voting_cypher() -> str:
    """
    Returns Rule nodes where at least one base token matches
    AND at least one expanded token matches.
    base_tokens must match to ensure relevance.
    all_tokens broadens the net for scoring.
    """
    return """
MATCH (z:Rule)
WHERE (
    ANY(word IN $base_tokens WHERE toLower(coalesce(z.action, "")) CONTAINS word)
    OR
    ANY(word IN $base_tokens WHERE toLower(coalesce(z.result, "")) CONTAINS word)
  )
  AND (
    ANY(word IN $all_tokens WHERE toLower(coalesce(z.action, "")) CONTAINS word)
    OR
    ANY(word IN $all_tokens WHERE toLower(coalesce(z.result, "")) CONTAINS word)
  )
RETURN
    z.rule_id  AS rule_id,
    z.type     AS type,
    z.action   AS action,
    z.result   AS result,
    z.art_ref  AS art_ref,
    z.reg_name AS reg_name
"""


def build_article_content_cypher() -> str:
    return """
MATCH (a:Article)
WHERE a.number IN $art_refs
RETURN
    a.number   AS art_ref,
    a.content  AS content
"""


# ========== 7) Vote Scoring ==========


def _compute_vote_score(
    row: dict[str, Any],
    base_tokens: list[str],
    all_tokens: list[str],
) -> int:
    """
    Count weighted token matches in rule's action + result fields.

    Base tokens (original question words) are worth BASE_TOKEN_WEIGHT (3x).
    Expanded tokens (synonyms/variations) are worth 1x.

    A token can score in both action and result fields.

    Example:
        base_tokens = ["minutes", "late", "barred", "exam"]
        Rule 4 action = "Students arriving more than 20 minutes..."
            "minutes" → base → weight=3 ✅
            "exam"    → base → weight=3 ✅
            total = 6  (+ expanded token matches on top)

        Article 19 rule = "late submission of grades"
            "late"    → base → weight=3 ✅
            total = 3
    """
    row_action = str(row.get("action", "") or "").lower()
    row_result = str(row.get("result", "") or "").lower()

    base_set = set(base_tokens)
    votes = 0

    for token in all_tokens:
        weight = BASE_TOKEN_WEIGHT if token in base_set else 1
        if token in row_action:
            votes += weight
        if token in row_result:
            votes += weight

    return votes


def _rank_by_votes(
    rows: list[dict[str, Any]],
    base_tokens: list[str],
    all_tokens: list[str],
) -> list[dict[str, Any]]:
    """
    Score each rule by weighted vote count, dedupe by rule_id, sort descending.
    """
    merged: dict[str, dict[str, Any]] = {}

    for row in rows:
        rule_id = str(row.get("rule_id", "") or "").strip()
        if not rule_id:
            continue
        row["vote_score"] = _compute_vote_score(row, base_tokens, all_tokens)
        if rule_id not in merged:
            merged[rule_id] = row
        elif row["vote_score"] > merged[rule_id]["vote_score"]:
            merged[rule_id] = row

    ranked = sorted(
        merged.values(),
        key=lambda x: x.get("vote_score", 0),
        reverse=True,
    )

    print("\n=== RULE VOTE RANKING ===")
    for idx, row in enumerate(ranked, start=1):
        print(
            f"[{idx}] rule_id={row.get('rule_id')} | "
            f"votes={row.get('vote_score')} | "
            f"art_ref={row.get('art_ref')} | "
            f"action={str(row.get('action', ''))[:60]}"
        )

    return ranked


# ========== 8) Article Aggregation ==========


def _aggregate_votes_by_article(
    ranked_rules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Aggregate rule vote scores up to the article level.
    Normalizes by rule count so articles with many weak rules
    don't outrank articles with few strong rules.

    normalized_score = total_votes / rule_count

    Example:
        Rule 4:    total=18  count=2  normalized=9.0  ← wins
        Article 3: total=36  count=17 normalized=2.1  ← drops
    """
    article_scores: dict[str, dict[str, Any]] = {}

    for rule in ranked_rules:
        art_ref = str(rule.get("art_ref", "") or "").strip()
        if not art_ref:
            continue

        vote_score = int(rule.get("vote_score", 0) or 0)

        if art_ref not in article_scores:
            article_scores[art_ref] = {
                "art_ref": art_ref,
                "total_votes": 0,
                "rule_count": 0,
            }

        article_scores[art_ref]["total_votes"] += vote_score
        article_scores[art_ref]["rule_count"] += 1

    # Normalize by rule count — penalizes articles that win only by volume
    for art in article_scores.values():
        art["normalized_score"] = art["total_votes"] / art["rule_count"]

    ranked_articles = sorted(
        article_scores.values(),
        key=lambda x: x["normalized_score"],
        reverse=True,
    )

    print("\n=== ARTICLE VOTE AGGREGATION ===")
    for idx, article in enumerate(ranked_articles, start=1):
        print(
            f"[{idx}] art_ref={article['art_ref']} | "
            f"total_votes={article['total_votes']} | "
            f"rules_matched={article['rule_count']} | "
            f"normalized={article['normalized_score']:.2f}"
        )

    return ranked_articles


# ========== 9) Article Content Fetcher ==========


def fetch_top_article_contents(
    ranked_articles: list[dict[str, Any]],
    top_n: int = 2,
) -> list[dict[str, Any]]:
    """
    Take the top_n articles by normalized score,
    fetch their full content from Neo4j.

    Returns list of dicts: [{art_ref, content}, ...]
    """
    if driver is None:
        return []

    art_refs = [a["art_ref"] for a in ranked_articles[:top_n] if a.get("art_ref")]

    if not art_refs:
        print("\n[article fetch] No art_refs to fetch.")
        return []

    print(f"\n=== FETCHING ARTICLE CONTENT ===")
    print(f"Fetching top {top_n} articles: {art_refs}")

    query = build_article_content_cypher()

    with driver.session() as session:
        records = session.run(query, {"art_refs": art_refs})
        articles = [dict(record) for record in records]

    print(f"Articles fetched: {len(articles)}")
    for a in articles:
        content_preview = str(a.get("content", "") or "")[:80]
        print(f"  art_ref={a.get('art_ref')} | preview: {content_preview}...")

    return articles


# ========== 10) Retrieval ==========


def get_relevant_articles(
    params: dict[str, Any],
    top_k: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Full retrieval pipeline:
    1. Run voting Cypher query against Rule nodes
    2. Rank rules by weighted vote score (base tokens = 3x weight)
    3. Aggregate rule votes up to article level
    4. Normalize article scores by rule count
    5. Fetch full content for top 2 articles

    Returns:
        ranked_rules     — top_k ranked Rule rows
        ranked_articles  — articles sorted by normalized score
        article_contents — full content of top 2 Article nodes
    """
    if driver is None:
        return [], [], []

    cypher_fetch_count = 0
    query = build_voting_cypher()
    base_tokens = params["base_tokens"]
    all_tokens = params["all_tokens"]

    print("\n=== RETRIEVAL PARAMS ===")
    print(f"type (classifier): {params['type']}")
    print(f"base_tokens: {base_tokens}")
    print(f"token count: {len(all_tokens)}")
    print(f"all_tokens : {all_tokens}")

    with driver.session() as session:
        records = session.run(query, params)
        rows = [dict(record) for record in records]
    cypher_fetch_count += 1
    print(f"\nCypher fetch #1 (Rule query) → {len(rows)} raw hits")

    # Rank rules by weighted vote score
    ranked_rules = _rank_by_votes(rows, base_tokens, all_tokens)

    # Aggregate and normalize votes at article level
    ranked_articles = _aggregate_votes_by_article(ranked_rules)

    # Fetch full content for top 2 articles
    article_contents = fetch_top_article_contents(ranked_articles, top_n=2)
    if article_contents:
        cypher_fetch_count += 1

    print(f"\n=== CYPHER FETCH SUMMARY ===")
    print(f"Total Cypher fetches this query: {cypher_fetch_count}")
    print(f"  #1 — Rule nodes query")
    if cypher_fetch_count >= 2:
        print(f"  #2 — Article content fetch (top 2 art_refs)")

    print(f"\nReturning top {top_k} rules, top 2 articles")

    return ranked_rules[:top_k], ranked_articles, article_contents


# ========== 11) Generation ==========


def _format_articles_for_generation(
    article_contents: list[dict[str, Any]],
) -> str:
    if not article_contents:
        return "No article content retrieved."

    lines = []
    for i, article in enumerate(article_contents, start=1):
        lines.append(
            f"Article {i} (art_ref: {article.get('art_ref', '')}):\n"
            f"{str(article.get('content', '') or '').strip()}\n"
        )

    return "\n---\n".join(lines)


def generate_text(messages: list[dict[str, str]], max_new_tokens: int = 220) -> str:
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


def generate_answer(
    question: str,
    article_contents: list[dict[str, Any]],
) -> str:
    """
    Generate final answer using full Article content from top 2 ranked articles.
    """
    articles_context = _format_articles_for_generation(article_contents)

    messages = [
        {
            "role": "system",
            "content": """
You are a careful regulation QA assistant.

- Answer the user's question using ONLY the retrieved article content provided.
- Do not invent facts.
- If the article gives a partial answer, explain the limitation.
- Do NOT default to "I don't know" if some relevant content exists.
- Only say "I don't know" if no relevant content exists at all.

Instructions:
- Prefer the most directly relevant article.
- If both articles are relevant, synthesize them briefly.
- Cite the supporting source at the end in this format:
  [Source: <art_ref>]
- If more than one source is used, include multiple citations.
- Be concise and direct.
""".strip(),
        },
        {
            "role": "user",
            "content": f"""
Question:
{question}

Retrieved article content:
{articles_context}

Write the final answer.
""".strip(),
        },
    ]

    return generate_text(messages, max_new_tokens=220).strip()


# ========== 12) Main Pipeline ==========


def answer_question(user_q: str) -> str:
    """
    Full pipeline:
    1. Classify type only (LLM — one word output)
    2. Tokenize raw question — stopwords removed (expanded set)
    3. Expand tokens — specific tokens first, generic last
    4. Check DB connectivity — fail fast if unavailable
    5. Voting Cypher query against Rule nodes
    6. Rank rules by weighted vote score (base tokens = 3x)
    7. Aggregate rule votes up to article level, normalize by rule count
    8. Fetch top 2 articles by normalized score
    9. Generate answer from raw question + full article content
    """

    # Step 1 — type only
    question_type = classify_type(user_q)

    # Step 2+3 — tokenize raw question + expand (specific tokens first)
    params = build_voting_params(
        user_q=user_q,
        question_type=question_type,
        max_total=MAX_EXPANDED_TOKENS,
    )

    # Step 4 — fail fast if DB unavailable
    if driver is None:
        return "Unable to answer: database is not connected."

    # Step 5+6+7+8 — retrieve, rank, aggregate, fetch
    ranked_rules, ranked_articles, article_contents = get_relevant_articles(
        params=params,
        top_k=10,
    )

    # Step 9 — no results found
    if not article_contents:
        return "No relevant articles were found for your question."

    # Step 10 — generate from raw question + full article content
    return generate_answer(
        question=user_q,
        article_contents=article_contents,
    )


# ========== 13) Entry Point ==========

if __name__ == "__main__":
    if driver is None:
        print("\n❌ Neo4j driver is NOT connected. Exiting.\n")
    else:
        load_local_llm()

        print("=" * 50)
        print("🎓 NCU Regulation Assistant")
        print("=" * 50)
        print(
            "💡 Try: 'How many minutes late can a student be before they are barred from the exam?'"
        )
        print("👉 Type 'exit' to quit.\n")

        while True:
            try:
                user_q = input("\nUser: ").strip()

                if not user_q:
                    continue

                if user_q.lower() in {"exit", "quit"}:
                    print("👋 Bye!")
                    break

                answer = answer_question(user_q)
                print(f"\nBot: {answer}")

            except KeyboardInterrupt:
                print("\n👋 Bye!")
                break
            except NotImplementedError as e:
                print(f"⚠️ {e}")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
