import json
import re
import os
from typing import Dict, Any
from dotenv import load_dotenv
from neo4j import GraphDatabase

# nltk for word variations
import nltk
from nltk.corpus import wordnet

# ── NEW: embedding support ──────────────────────────────────────────────────
import numpy as np
from sentence_transformers import SentenceTransformer
# ────────────────────────────────────────────────────────────────────────────

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


# ── NEW: load embedding model once at startup ───────────────────────────────
# ⚠️  Change this to whatever model you used when building the DB embeddings.
#     It MUST be the same model — mismatched models produce meaningless scores.
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the embedding model (loaded once, reused for every query)."""
    global _embedding_model
    if _embedding_model is None:
        print(f"[embedding] Loading model: {EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("[embedding] Model loaded ✅")
    return _embedding_model


def embed_query(text: str) -> np.ndarray:
    """Return a unit-normalised embedding vector for text."""
    model = get_embedding_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.astype(np.float32)


def cosine_similarity(a: np.ndarray | list, b: np.ndarray | list) -> float:
    """
    Cosine similarity between two vectors.
    Embeddings stored in Neo4j come back as plain Python lists —
    this handles both lists and numpy arrays.
    Returns a float in [-1, 1]; 1.0 = identical direction.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
# ────────────────────────────────────────────────────────────────────────────


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
    "the", "a", "an", "and", "or", "of", "to", "from", "than", "more",
    "less", "be", "is", "are", "was", "were", "will", "shall", "should",
    "can", "could",
    # pronouns
    "my", "me", "i", "we", "us", "you", "he", "she", "they", "them",
    "their", "his", "her", "its",
    # question words
    "do", "how", "what", "when", "where", "why", "who", "which",
    # prepositions
    "if", "for", "in", "on", "at", "by", "with", "about", "this", "that",
    "it", "not", "no", "yes",
    # common verbs with no search value
    "get", "got", "have", "has", "had", "been",
    # quantifiers and determiners
    "any", "all", "some", "would", "may", "might", "also", "many", "much",
    "few", "each", "such", "per",
    # time/order words too generic to search
    "before", "after", "during", "while", "until", "since", "these",
    "those", "then", "than", "just", "only",
    # single letters and noise
    "x",
}

# Generic domain words — appear in almost every rule.
LOW_PRIORITY_TOKENS = {
    "student", "students", "exam", "exams", "course", "courses",
    "university", "ncu", "school", "semester", "academic", "study",
    "studies", "rule", "rules", "regulation", "regulations", "article",
    "department", "program", "degree",
}

EXPANSIONS_PER_TOKEN = 5
BASE_TOKEN_WEIGHT    = 4
LOW_PRIORITY_WEIGHT  = 1
MIN_BASE_TOKEN_MATCHES = 2

# ── NEW: embedding scoring weights ──────────────────────────────────────────
# How much each embedding field contributes to the combined embedding score.
# combined_embedding is holistic (action+result together), so it gets most weight.
EMBED_WEIGHT_COMBINED = 0.50   # combined_embedding vs query
EMBED_WEIGHT_ACTION   = 0.30   # action_embedding   vs query
EMBED_WEIGHT_RESULT   = 0.20   # result_embedding   vs query

# How to blend vote score vs embedding score in the final ranking.
# Increase EMBED_BLEND to trust embeddings more; decrease to trust keyword votes more.
VOTE_BLEND   = 0.55   # weight given to normalised keyword vote score
EMBED_BLEND  = 0.45   # weight given to embedding similarity score
# ────────────────────────────────────────────────────────────────────────────


# ========== 2) Helpers ==========

def count_words(text: str) -> int:
    if not text or not text.strip():
        return 0
    return len(re.findall(r"\b[\w'-]+\b", text.strip()))


def tokenize_for_retrieval(text: str) -> list[str]:
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
    return json.loads(response_text.strip())


def classify_type(user_input: str) -> str:
    prompt = build_type_classification_prompt(user_input)
    response_text = call_llm_once(prompt)


    try:
        data = parse_json_text(response_text)
        type_value = str(data.get("type", "")).strip().lower()
        if type_value not in ALLOWED_TYPES:
            print(f"[classify_type] Invalid type '{type_value}', falling back to 'other'")
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
    return f"""You are a search-term expansion assistant for a university academic regulation retrieval system.

Your job is to help match a search keyword to the exact words that appear in official university regulations, student handbooks, and academic policy documents.

For the word "{token}", generate exactly {EXPANSIONS_PER_TOKEN} alternative words or short phrases that:
- Are likely to appear in formal university regulation text (not casual speech)
- Include common legal/regulatory variants (e.g. plural/singular, verb/noun forms, passive forms)
- Include synonyms used in academic policy documents (e.g. "dismissed" → "expelled", "barred", "excluded")
- Are single words only (no multi-word phrases)

Rules:
- Return a flat JSON array of exactly {EXPANSIONS_PER_TOKEN} strings
- All words must be lowercase
- No duplicates
- No explanations, no markdown, no extra text
- Do NOT include the original word "{token}"
- Prioritize words found in legal/regulatory writing over everyday synonyms

Examples of good expansions:
  "late"     → ["tardy", "delayed", "overdue", "delinquent", "untimely"]
  "barred"   → ["excluded", "prohibited", "disqualified", "banned", "denied"]
  "cheating" → ["misconduct", "dishonesty", "plagiarism", "violation", "fraud"]
  "absent"   → ["missing", "nonattendance", "truancy", "unexcused", "default"]
  "penalty"  → ["sanction", "consequence", "deduction", "punishment", "forfeiture"]

Now expand: "{token}"
Output (JSON array only):"""


def _parse_llm_expansion(response_text: str, token: str) -> list[str]:
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
            parsed = re.findall(r"[\w'-]+", raw)
        return [
            str(w).strip().lower()
            for w in parsed
            if isinstance(w, str)
            and len(w.strip()) > 2
            and w.strip().lower() != token.lower()
            and w.strip().lower() not in STOPWORDS
        ]
    except (json.JSONDecodeError, TypeError) as e:
        print(f"[expansion] Parse error: {e} | text: {response_text[:100]}")
        return []


def _expand_token_with_llm(token: str, max_words: int) -> list[str]:
    ensure_llm_loaded()
    prompt = _build_synonym_prompt(token)
    tok = get_tokenizer()
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    llm = get_raw_pipeline()
    raw = llm(formatted_prompt, max_new_tokens=100)
    response_text = (
        raw[0].get("generated_text", "") if isinstance(raw, list) and raw else str(raw)
    )
    return _parse_llm_expansion(response_text, token)[:max_words]


def expand_tokens(tokens: list[str]) -> list[str]:
    print(f"\n=== TOKEN EXPANSION (per_token_cap={EXPANSIONS_PER_TOKEN}) ===")
    print(f"Original tokens: {tokens}")

    expanded_set = set(tokens)
    ordered = list(tokens)

    for token in tokens:
        if token in LOW_PRIORITY_TOKENS:
            print(f"\n[skip expand] '{token}' is a low-priority domain word — not expanding")
            continue

        print(f"\n[expanding] '{token}'")

        wn_variations = _get_wordnet_variations(token)
        wn_filtered   = [w for w in wn_variations if w not in expanded_set and w not in STOPWORDS]
        wn_take        = wn_filtered[:EXPANSIONS_PER_TOKEN]
        slots_remaining = EXPANSIONS_PER_TOKEN - len(wn_take)

        print(f"  [WordNet] '{token}' → {wn_take} ({len(wn_take)} taken, {slots_remaining} slots left for LLM)")
        for word in wn_take:
            expanded_set.add(word)
            ordered.append(word)

        if slots_remaining > 0:
            llm_synonyms = _expand_token_with_llm(token, max_words=slots_remaining)
            llm_filtered = [w for w in llm_synonyms if w not in expanded_set and w not in STOPWORDS]
            llm_take     = llm_filtered[:slots_remaining]
            print(f"  [LLM]     '{token}' → {llm_take} ({len(llm_take)} taken)")
            for word in llm_take:
                expanded_set.add(word)
                ordered.append(word)
        else:
            print(f"  [LLM]     skipped — WordNet already filled {EXPANSIONS_PER_TOKEN} slots")

    print(f"\n=== EXPANSION RESULT ===")
    print(f"Original count : {len(tokens)}")
    print(f"Total count    : {len(ordered)}")
    print(f"Base tokens    : {tokens}")
    print(f"All tokens     : {ordered}")

    return ordered


# ========== 5) Params Builder ==========

def build_voting_params(
    user_q: str,
    question_type: str,
) -> dict[str, Any]:
    base_tokens = tokenize_for_retrieval(user_q)

    print("\n=== TOKEN DEBUG ===")
    print("Raw question :", user_q)
    print("Base tokens  :", base_tokens)

    expanded_tokens = expand_tokens(base_tokens)

    # ── NEW: embed the raw question once here ───────────────────────────────
    print("\n[embedding] Embedding query...")
    query_embedding = embed_query(user_q)
    print(f"[embedding] Query vector shape: {query_embedding.shape}")
    # ────────────────────────────────────────────────────────────────────────

    return {
        "type": question_type,
        "base_tokens": base_tokens,
        "all_tokens": expanded_tokens,
        "query_embedding": query_embedding,   # ← NEW
    }


# ========== 6) Cypher Queries ==========

def build_voting_cypher() -> str:
    """
    Same as before, but now also returns the three embedding fields so
    Python-side cosine similarity can be computed without a second round-trip.

    Note: embedding properties are returned as plain Python lists by the
    Neo4j driver — numpy handles those fine in cosine_similarity().
    """
    return """
MATCH (z:Rule)
WITH z,
  size([
    word IN $base_tokens
    WHERE toLower(coalesce(z.action, '')) =~ ('.*\\\\b' + word + '\\\\b.*')
       OR toLower(coalesce(z.result, '')) =~ ('.*\\\\b' + word + '\\\\b.*')
  ]) AS base_hit_count
WHERE base_hit_count >= $min_base_matches
RETURN
    z.rule_id            AS rule_id,
    z.type               AS type,
    z.action             AS action,
    z.result             AS result,
    z.art_ref            AS art_ref,
    z.reg_name           AS reg_name,
    base_hit_count,
    z.action_embedding   AS action_embedding,
    z.result_embedding   AS result_embedding,
    z.combined_embedding AS combined_embedding
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
    row_action = str(row.get("action", "") or "").lower()
    row_result = str(row.get("result", "") or "").lower()
    base_set   = set(base_tokens)
    votes      = 0

    for token in all_tokens:
        pattern = r"\b" + re.escape(token) + r"\b"
        if token in base_set:
            weight = BASE_TOKEN_WEIGHT
        elif token in LOW_PRIORITY_TOKENS:
            weight = LOW_PRIORITY_WEIGHT
        else:
            weight = 1

        if re.search(pattern, row_action):
            votes += weight
        if re.search(pattern, row_result):
            votes += weight

    return votes


# ── NEW: embedding score for a single rule row ──────────────────────────────
def _compute_embedding_score(
    row: dict[str, Any],
    query_embedding: np.ndarray,
) -> float:
    """
    Weighted cosine similarity across all three embedding fields.

    Fields that are missing or empty (None / []) silently contribute 0
    so that partially-populated rules are penalised but not crashed.

    Returns a float in [0, 1] (embeddings are unit-normalised, so cosine
    similarity is in [-1, 1]; negative values are clamped to 0 since a
    negative similarity is practically meaningless for retrieval).
    """
    def _safe_sim(field_key: str, weight: float) -> float:
        vec = row.get(field_key)
        if not vec:          # None or empty list
            return 0.0
        sim = cosine_similarity(query_embedding, vec)
        return max(0.0, sim) * weight   # clamp negatives to 0

    score = (
        _safe_sim("action_embedding",   EMBED_WEIGHT_ACTION)
        + _safe_sim("result_embedding",   EMBED_WEIGHT_RESULT)
        + _safe_sim("combined_embedding", EMBED_WEIGHT_COMBINED)
    )
    return score   # range: [0, 1] when weights sum to 1
# ────────────────────────────────────────────────────────────────────────────


def _rank_by_votes(
    rows: list[dict[str, Any]],
    base_tokens: list[str],
    all_tokens: list[str],
    query_embedding: np.ndarray,   # ← NEW param
) -> list[dict[str, Any]]:
    """
    Score each rule by a BLENDED score:
        final_score = VOTE_BLEND  * normalised_vote_score
                    + EMBED_BLEND * embedding_score

    Vote scores are an unbounded integer — they are normalised to [0, 1]
    by dividing by the maximum vote score in the candidate set before blending.
    Embedding scores are already in [0, 1].

    Deduplication by rule_id keeps the highest-scoring record per rule.
    """
    merged: dict[str, dict[str, Any]] = {}

    for row in rows:
        rule_id = str(row.get("rule_id", "") or "").strip()
        if not rule_id:
            continue

        # ── keyword vote score (raw integer) ──────────────────────────────
        vote_score = _compute_vote_score(row, base_tokens, all_tokens)
        row["vote_score"] = vote_score

        # ── NEW: embedding similarity score ───────────────────────────────
        embedding_score = _compute_embedding_score(row, query_embedding)
        row["embedding_score"] = embedding_score
        # ──────────────────────────────────────────────────────────────────

        if rule_id not in merged:
            merged[rule_id] = row
        elif vote_score > merged[rule_id]["vote_score"]:
            merged[rule_id] = row

    # ── NEW: normalise vote scores to [0, 1] before blending ─────────────
    all_rows = list(merged.values())
    max_vote  = max((r["vote_score"] for r in all_rows), default=1) or 1
    for row in all_rows:
        norm_vote           = row["vote_score"] / max_vote
        row["norm_vote"]    = norm_vote
        row["final_score"]  = (
            VOTE_BLEND  * norm_vote
            + EMBED_BLEND * row["embedding_score"]
        )
    # ─────────────────────────────────────────────────────────────────────

    ranked = sorted(
        all_rows,
        key=lambda x: x.get("final_score", 0.0),
        reverse=True,
    )

    

    return ranked


# ========== 8) Article Aggregation ==========

def _aggregate_votes_by_article(
    ranked_rules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Aggregate the blended final_score up to the article level.
    Uses the same normalised-by-count logic as before, but now on final_score
    instead of raw vote_score — so embedding signal is already baked in.
    """
    article_scores: dict[str, dict[str, Any]] = {}

    for rule in ranked_rules:
        art_ref = str(rule.get("art_ref", "") or "").strip()
        if not art_ref:
            continue

        # ── NEW: use final_score (blended) instead of raw vote_score ──────
        final_score = float(rule.get("final_score", 0.0) or 0.0)
        # ──────────────────────────────────────────────────────────────────

        if art_ref not in article_scores:
            article_scores[art_ref] = {
                "art_ref": art_ref,
                "total_score": 0.0,
                "rule_count": 0,
            }

        article_scores[art_ref]["total_score"] += final_score
        article_scores[art_ref]["rule_count"]  += 1

    for art in article_scores.values():
        art["normalized_score"] = art["total_score"] / art["rule_count"]

    ranked_articles = sorted(
        article_scores.values(),
        key=lambda x: x["normalized_score"],
        reverse=True,
    )

    

    return ranked_articles


# ========== 9) Article Content Fetcher ==========

def fetch_top_article_contents(
    ranked_articles: list[dict[str, Any]],
    top_n: int = 4 ,
) -> list[dict[str, Any]]:
    if driver is None:
        return []

    art_refs = [a["art_ref"] for a in ranked_articles[:top_n] if a.get("art_ref")]
    if not art_refs:
        print("\n[article fetch] No art_refs to fetch.")
        return []

    

    query = build_article_content_cypher()
    with driver.session() as session:
        records = session.run(query, {"art_refs": art_refs})
        articles = [dict(record) for record in records]

    
    
    return articles


# ========== 10) Retrieval ==========

def get_relevant_articles(
    params: dict[str, Any],
    top_k: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Full retrieval pipeline:
    1. Run voting Cypher — fetches rule fields + all three embedding vectors
    2. Score each rule:
         a. keyword vote score  (whole-word regex, weighted)
         b. embedding score     (weighted cosine sim across action/result/combined)
         c. final_score         = VOTE_BLEND * norm_vote + EMBED_BLEND * embed_sim
    3. Deduplicate by rule_id, sort by final_score
    4. Aggregate final_score up to article level, normalise by rule count
    5. Fetch full content for top 2 articles
    """
    if driver is None:
        return [], [], []

    cypher_fetch_count = 0
    query        = build_voting_cypher()
    base_tokens  = params["base_tokens"]
    all_tokens   = params["all_tokens"]
    query_embedding = params["query_embedding"]   # ← NEW

    print("\n=== RETRIEVAL PARAMS ===")
    print(f"type (classifier)   : {params['type']}")
    print(f"base_tokens         : {base_tokens}")
    print(f"all_tokens count    : {len(all_tokens)}")
    print(f"all_tokens          : {all_tokens}")
    print(f"min_base_matches    : {MIN_BASE_TOKEN_MATCHES}")
    print(f"query_embedding dim : {query_embedding.shape[0]}")   # ← NEW
    print(f"vote/embed blend    : {VOTE_BLEND}/{EMBED_BLEND}")    # ← NEW

    with driver.session() as session:
        records = session.run(
            query,
            {
                "base_tokens": base_tokens,
                "all_tokens": all_tokens,
                "min_base_matches": MIN_BASE_TOKEN_MATCHES,
            },
        )
        rows = [dict(record) for record in records]
    cypher_fetch_count += 1
    print(f"\nCypher fetch #1 → {len(rows)} raw hits (base_hit_count >= {MIN_BASE_TOKEN_MATCHES})")

    # Rank rules by blended score
    ranked_rules = _rank_by_votes(rows, base_tokens, all_tokens, query_embedding)   # ← NEW arg

    # Aggregate blended scores up to article level
    ranked_articles = _aggregate_votes_by_article(ranked_rules)

    # Fetch full content for top 2 articles
    article_contents = fetch_top_article_contents(ranked_articles, top_n=4)
    if article_contents:
        cypher_fetch_count += 1

    print(f"\n=== CYPHER FETCH SUMMARY ===")
    print(f"Total Cypher fetches this query: {cypher_fetch_count}")
    print(f"  #1 — Rule nodes query (whole-word regex + embeddings returned)")
    if cypher_fetch_count >= 2:
        print(f"  #2 — Article content fetch (top 4 art_refs)")

    
    return ranked_rules[:top_k], ranked_articles, article_contents


# ========== 11) Generation ==========

def _format_articles_for_generation(article_contents: list[dict[str, Any]]) -> str:
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
    tok  = get_tokenizer()
    pipe = get_raw_pipeline()

    if tok is None or pipe is None:
        load_local_llm()
        tok  = get_tokenizer()
        pipe = get_raw_pipeline()

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return pipe(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"].strip()


def generate_answer(question: str, article_contents: list[dict[str, Any]]) -> str:
    articles_context = _format_articles_for_generation(article_contents)

    # Build the list of valid art_refs to explicitly constrain citation
    valid_refs = [a.get("art_ref", "") for a in article_contents]
    valid_refs_str = ", ".join(valid_refs)

    messages = [
        {
            "role": "system",
            "content": f"""
You are a careful regulation QA assistant.

STRICT RULES:
- Use ONLY the article content provided below. Do not use any outside knowledge.
- You may ONLY cite these sources: {valid_refs_str}
- If you cannot find the answer in the provided articles, say: "The provided articles do not contain a clear answer to this question."
- NEVER invent facts, numbers, or article references not present in the content below.
- A wrong answer with a fake citation is worse than admitting the answer is not found.

Instructions:
- Cite the supporting source at the end: [Source: <art_ref>]
- Be concise and direct. Short answer, do not over explain.
""".strip(),
        },
        {
            "role": "user",
            "content": f"""
Question:
{question}

Retrieved article content:
{articles_context}

Write the final answer. Only use content from the articles above.
""".strip(),
        },
    ]

    return generate_text(messages, max_new_tokens=220).strip()

# ========== 12) Main Pipeline ==========

def answer_question(user_q: str) -> str:
    """
    Full pipeline:
    1. Classify type (LLM)
    2. Tokenize + expand tokens (WordNet + LLM)
    3. Embed the raw query                             ← NEW
    4. Voting Cypher — returns rules + embedding vecs  ← UPDATED
    5. Score: keyword votes + cosine similarity        ← NEW
    6. Blend into final_score, rank rules              ← NEW
    7. Aggregate final_score to article level
    8. Fetch top 2 article contents
    9. Generate answer
    """
    question_type = classify_type(user_q)

    params = build_voting_params(
        user_q=user_q,
        question_type=question_type,
    )

    if driver is None:
        return "Unable to answer: database is not connected."

    ranked_rules, ranked_articles, article_contents = get_relevant_articles(
        params=params,
        top_k=10,
    )

    if not article_contents:
        return "No relevant articles were found for your question."

    return generate_answer(question=user_q, article_contents=article_contents)


# ========== 13) Entry Point ==========

if __name__ == "__main__":
    if driver is None:
        print("\n❌ Neo4j driver is NOT connected. Exiting.\n")
    else:
        load_local_llm()
        get_embedding_model()   # ← NEW: warm up embedding model at startup

        print("=" * 50)
        print("🎓 NCU Regulation Assistant")
        print("=" * 50)
        print("💡 Try: 'How many minutes late can a student be before they are barred from the exam?'")
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