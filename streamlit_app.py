import os
import json
import itertools
import streamlit as st
from openai import OpenAI

MODEL = "gpt-4.1"
MODEL_MINI = "gpt-4.1-mini"
TOOLS = [{"type": "web_search_preview"}]

developer_message = """
You are an expert Deep Researcher.
You provide complete and in depth research to the user.
"""


def get_openai_client(api_key: str):
    """Return an OpenAI client instance configured with the given key."""
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI()


def ask_clarifying_questions(client: OpenAI, topic: str):
    prompt = f"""
Ask 5 numbered clarifying questions about the topic of research: {topic}.
The goal of the questions is to understand the intented purpose of the research.
Reply only with the questions.
"""
    clarify = client.responses.create(
        model=MODEL_MINI,
        input=prompt,
        instructions=developer_message,
    )
    questions = [q.strip()
                 for q in clarify.output[0].content[0].text.split("\n") if q.strip()]
    return questions, clarify.id


def generate_plan(client: OpenAI, topic: str, questions: list[str], answers: list[str], prev_id: str):
    prompt = f"""
Using the user answers {answers} to the {questions}, write a goal sentence and 5 web searches queries for the research about {topic}
Output: A json list of the goal and the 5 web queries that will reach it.
Format: {{\"goal\": \"...\", \"queries\": [\"q1\", ....]}}
"""
    goal_and_queries = client.responses.create(
        model=MODEL,
        input=prompt,
        previous_response_id=prev_id,
        instructions=developer_message,
    )
    plan = json.loads(goal_and_queries.output[0].content[0].text)
    return plan["goal"], plan["queries"], goal_and_queries.id


def run_search(client: OpenAI, query: str, prev_id: str):
    web_search = client.responses.create(
        model=MODEL,
        input=f"search: {query}",
        previous_response_id=prev_id,
        instructions=developer_message,
        tools=TOOLS,
    )
    return {"query": query, "resp_id": web_search.output[1].content[0].text}


def evaluate(client: OpenAI, collected: list[dict], goal: str):
    review = client.responses.create(
        model=MODEL,
        input=[
            {"role": "developer", "content": f"Research goal: {goal}"},
            {"role": "assistant", "content": json.dumps(collected)},
            {"role": "user", "content": "Does this information answer the research goal? Answer Yes or No only"},
        ],
        instructions=developer_message,
    )
    return "yes" in review.output[0].content[0].text.lower()


def perform_research(client: OpenAI, goal: str, queries: list[str], prev_id: str):
    collected = []
    for _ in itertools.count():
        for q in queries:
            collected.append(run_search(client, q, prev_id))
        if evaluate(client, collected, goal):
            break
        more_searches = client.responses.create(
            model=MODEL,
            input=[
                {"role": "assistant",
                    "content": f"Current data: {json.dumps(collected)}"},
                {"role": "user", "content": f"This has not met the goal: {goal}. Write 5 other web searchs to achieve the goal"},
            ],
            instructions=developer_message,
            previous_response_id=prev_id,
        )
        queries = json.loads(more_searches.output[0].content[0].text)
        collected = []
    return collected


def generate_report(client: OpenAI, goal: str, collected: list[dict]):
    report = client.responses.create(
        model=MODEL,
        input=[
            {"role": "developer", "content": (
                f"Write a complete and detail report about research goal: {goal} "
                "Cite Sources inline using [n] and append a reference "
                "list mapping [n] to url"
            )},
            {"role": "assistant", "content": json.dumps(collected)},
        ],
        instructions=developer_message,
    )
    return report.output[0].content[0].text


def main():
    st.title("Deep Research Assistant")

    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if not api_key:
        st.info("Please enter your OpenAI API key to begin.")
        st.stop()

    topic = st.text_input("Enter the topic of research:")
    if not topic:
        st.stop()

    if st.button("Generate Clarifying Questions"):
        client = get_openai_client(api_key)
        questions, clarify_id = ask_clarifying_questions(client, topic)
        st.session_state["questions"] = questions
        st.session_state["clarify_id"] = clarify_id
        st.session_state["answers"] = [""] * len(questions)

    if "questions" in st.session_state:
        st.subheader("Clarifying Questions")
        answers = []
        for idx, q in enumerate(st.session_state["questions"]):
            ans = st.text_input(q, key=f"answer_{idx}")
            answers.append(ans)
        if st.button("Run Deep Research"):
            if "" in answers:
                st.warning(
                    "Please answer all clarifying questions before proceeding.")
                st.stop()
            client = get_openai_client(api_key)
            goal, queries, goal_id = generate_plan(
                client, topic, st.session_state["questions"], answers, st.session_state["clarify_id"]
            )
            st.success(f"Research goal: {goal}")
            with st.spinner("Performing web research. This may take a while..."):
                collected = perform_research(client, goal, queries, goal_id)
            with st.spinner("Generating final report..."):
                report_md = generate_report(client, goal, collected)
            st.markdown("## Final Report")
            st.markdown(report_md)


if __name__ == "__main__":
    main()
