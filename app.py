import os
import json
import itertools
import streamlit as st
from openai import OpenAI

# Models and tools identical to original script
MODEL = "gpt-4.1"
MODEL_MINI = "gpt-4.1-mini"
TOOLS = [{"type": "web_search_preview"}]

# System / developer message
DEVELOPER_MESSAGE = (
    "You are an expert Deep Researcher.\n"
    "You provide complete and in depth research to the user."
)

# ----------------------------- Helper Functions ----------------------------- #


def get_openai_client(api_key: str) -> OpenAI:
    """Return OpenAI client with environment key set."""
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI()


def ask_clarifying_questions(client: OpenAI, topic: str):
    """Return 5 clarifying questions and the response id used to generate them."""
    prompt = (
        f"Ask 5 numbered clarifying questions about the topic of research: {topic}. "
        "The goal of the questions is to understand the intented purpose of the research. "
        "Reply only with the questions."
    )
    clarify = client.responses.create(
        model=MODEL_MINI,
        input=prompt,
        instructions=DEVELOPER_MESSAGE,
    )
    questions = [q.strip()
                 for q in clarify.output[0].content[0].text.split("\n") if q.strip()]
    return questions, clarify.id


def create_plan(client: OpenAI, topic: str, questions: list[str], answers: list[str], prev_id: str):
    """Generate goal sentence and 5 web search queries."""
    prompt = (
        f"Using the user answers {answers} to the {questions}, write a goal sentence and 5 web searches "
        f"queries for the research about {topic} \n"
        "Output: A json list of the goal and the 5 web queries that will reach it.\n"
        "Format: {\"goal\": \"...\", \"queries\": [\"q1\", ....]}"
    )
    resp = client.responses.create(
        model=MODEL,
        input=prompt,
        previous_response_id=prev_id,
        instructions=DEVELOPER_MESSAGE,
    )
    plan = json.loads(resp.output[0].content[0].text)
    return plan["goal"], plan["queries"], resp.id


def run_search(client: OpenAI, query: str, prev_id: str):
    """Perform a single web search query and return record with ids."""
    search_resp = client.responses.create(
        model=MODEL,
        input=f"search: {query}",
        previous_response_id=prev_id,
        instructions=DEVELOPER_MESSAGE,
        tools=TOOLS,
    )
    return {"query": query, "resp_id": search_resp.output[1].content[0].text}


def evaluate_progress(client: OpenAI, goal: str, collected: list[dict]) -> bool:
    """Return True if current collected data satisfies the goal."""
    review = client.responses.create(
        model=MODEL,
        input=[
            {"role": "developer", "content": f"Research goal: {goal}"},
            {"role": "assistant", "content": json.dumps(collected)},
            {"role": "user", "content": "Does this information answer the research goal? Answer Yes or No only"},
        ],
        instructions=DEVELOPER_MESSAGE,
    )
    return "yes" in review.output[0].content[0].text.lower()


def conduct_research(client: OpenAI, goal: str, queries: list[str], prev_id: str):
    """Iteratively search until goal is satisfied, mirroring notebook flow."""
    collected = []
    for _ in itertools.count():
        for q in queries:
            collected.append(run_search(client, q, prev_id))
        if evaluate_progress(client, goal, collected):
            break
        # Need more data → ask LLM for 5 alternative queries
        more = client.responses.create(
            model=MODEL,
            input=[
                {"role": "assistant",
                    "content": f"Current data: {json.dumps(collected)}"},
                {"role": "user", "content": f"This has not met the goal: {goal}. Write 5 other web searchs to achieve the goal"},
            ],
            instructions=DEVELOPER_MESSAGE,
            previous_response_id=prev_id,
        )
        queries = json.loads(more.output[0].content[0].text)
        collected = []  # reset and try again
    return collected


def generate_final_report(client: OpenAI, goal: str, collected: list[dict]):
    """Ask LLM to compile the final deep-research report."""
    report = client.responses.create(
        model=MODEL,
        input=[
            {"role": "developer", "content": (
                f"Write a complete and detail report about research goal: {goal} "
                "Cite Sources inline using [n] and append a reference list mapping [n] to url"
            )},
            {"role": "assistant", "content": json.dumps(collected)},
        ],
        instructions=DEVELOPER_MESSAGE,
    )
    return report.output[0].content[0].text

# ------------------------------ Streamlit UI ------------------------------ #


def main():
    st.set_page_config(page_title="Deep Research Assistant", layout="wide")
    st.title("🔍 Deep Research Assistant")

    # Get API key from Streamlit secrets (if present) or environment variable
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error(
            "OPENAI_API_KEY not found. Set it in Streamlit Secrets or as an environment variable.")
        st.stop()

    # Topic input
    topic = st.text_input("Topic of research",
                          placeholder="e.g., Quantum Computing for Beginners")
    if not topic:
        st.stop()

    # Section: Clarifying Questions
    if st.button("Generate Clarifying Questions") or "questions" in st.session_state:
        client = get_openai_client(api_key)

        if "questions" not in st.session_state:
            # fresh generation
            qs, clarify_id = ask_clarifying_questions(client, topic)
            st.session_state.update(
                {"questions": qs, "clarify_id": clarify_id})

        st.subheader("Please answer the following questions:")
        answers = []
        for idx, q in enumerate(st.session_state["questions"]):
            ans = st.text_input(q, key=f"answer_{idx}")
            answers.append(ans)

        if st.button("Run Deep Research"):
            if any(a.strip() == "" for a in answers):
                st.warning("Answer all questions before proceeding.")
                st.stop()

            # Planning phase
            goal, queries, goal_id = create_plan(
                client, topic, st.session_state["questions"], answers, st.session_state["clarify_id"]
            )
            st.success(f"Research Goal: {goal}")

            # Research phase
            with st.spinner("Gathering information from the web..."):
                collected = conduct_research(client, goal, queries, goal_id)

            # Report phase
            with st.spinner("Generating final report"):
                report_md = generate_final_report(client, goal, collected)

            st.markdown("## 📄 Final Report")
            st.markdown(report_md)


if __name__ == "__main__":
    main()
