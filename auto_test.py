# auto_test.py
import os
import sys
from pathlib import Path

for key in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    if key in os.environ:
        del os.environ[key]

import json
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT_DIR = Path(__file__).resolve().parent
TEST_DATA_PATH = ROOT_DIR / "test_data.json"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# answer_question runs the full pipeline internally:
# classify_type → build_voting_params → get_relevant_articles → generate_answer
from query_system import answer_question, generate_text

load_dotenv()


def preflight_checks() -> bool:
    """Fail fast when grading environment is not ready."""
    if not (ROOT_DIR / "query_system.py").exists():
        print(f"[X] Error: query_system.py not found in {ROOT_DIR}")
        return False

    if not TEST_DATA_PATH.exists():
        print(f"[X] Error: test data not found: {TEST_DATA_PATH}")
        return False

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))

    try:
        driver = GraphDatabase.driver(uri, auth=auth)
        driver.verify_connectivity()
        with driver.session() as session:
            count = session.run("MATCH (r:Rule) RETURN count(r) AS c").single()["c"]
        driver.close()
    except Exception as e:
        print(f"[X] Error: Neo4j preflight failed: {e}")
        print("    Hint: Start Neo4j and run build_kg.py before auto-test.")
        return False

    if count == 0:
        print(
            "[X] Error: Neo4j has 0 Rule nodes. Please run setup_data.py and build_kg.py first."
        )
        return False

    print(f"[OK] Preflight passed: Neo4j connected, Rule nodes = {count}")
    return True


def ask_bot(question: str) -> str:
    """
    Run the full QA pipeline for one question.
    answer_question() handles everything:
        classify_type → build_voting_params → get_relevant_articles → generate_answer
    """
    try:
        return answer_question(user_q=question)
    except Exception as e:
        return f"Error: {str(e)}"


def evaluate_with_llm(question: str, expected: str, actual: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an impartial judge evaluating a Q&A system for university regulations. "
                "Respond with exactly one word: PASS or FAIL."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Expected Answer: {expected}\n"
                f"Actual Answer from Bot: {actual}\n\n"
                "Does the Actual Answer convey the same key information as the Expected Answer?\n"
                "Rules:\n"
                "1. If the bot says it cannot find information or gives a wrong number/fact -> FAIL.\n"
                "2. Minor wording differences (e.g. '20 minutes' vs 'twenty minutes') -> PASS.\n"
                "3. More detail than expected but the core fact is correct -> PASS.\n"
                "Answer with one word only: PASS or FAIL."
            ),
        },
    ]

    try:
        text = generate_text(messages).strip()
        upper_text = text.upper()

        if "FAIL" in upper_text and "PASS" not in upper_text:
            return "FAIL"
        if "PASS" in upper_text and "FAIL" not in upper_text:
            return "PASS"

        # fallback: inspect last non-empty line
        lines = [line.strip().upper() for line in text.splitlines() if line.strip()]
        if lines:
            last = lines[-1]
            if "PASS" in last:
                return "PASS"
            if "FAIL" in last:
                return "FAIL"

        return "FAIL"

    except Exception as e:
        return f"FAIL (Judge Error: {str(e)})"


def run_llm_evaluation():
    if not preflight_checks():
        return

    try:
        with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print("[X] Error: test_data.json not found!")
        return

    print(f"[*] Starting Evaluation for {len(test_cases)} Questions...\n")

    passed_count = 0
    results_log = []

    for case in test_cases:
        qid = case["id"]
        question = case["question"]
        expected_answer = case["answer"]

        print(f"Testing Q{qid}: {question}")

        start_time = time.time()
        bot_answer = ask_bot(question)
        verdict = evaluate_with_llm(question, expected_answer, bot_answer)
        duration = time.time() - start_time

        status_icon = "[OK]" if "PASS" in verdict else "[FAIL]"
        if "PASS" in verdict:
            passed_count += 1

        print(f"  -> Bot Says: {bot_answer.strip()}")
        print(f"  -> Judge: {status_icon} {verdict} (Time: {duration:.2f}s)")
        print("-" * 50)

        results_log.append(
            {
                "id": qid,
                "question": question,
                "expected": expected_answer,
                "bot_response": bot_answer,
                "result": verdict,
            }
        )

    print("\n" + "=" * 30)
    print("=== Evaluation Summary ===")
    print(f"Total  : {len(test_cases)}")
    print(f"Passed : {passed_count}")
    print(f"Failed : {len(test_cases) - passed_count}")
    if len(test_cases) > 0:
        print(f"Accuracy: {(passed_count / len(test_cases)) * 100:.1f}%")
    print("=" * 30)


if __name__ == "__main__":
    run_llm_evaluation()
