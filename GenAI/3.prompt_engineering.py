"""
================================================================================
 PROMPT ENGINEERING — COMPLETE, RUNNABLE CODE REFERENCE
 AI Bees Academy · Session 2 of 2 · Customer Support Ticket Triage
================================================================================

Runnable LangChain + Vertex AI (Gemini) examples for every technique covered in
Session 2. The SAME sample support ticket is reused across every technique so
you can compare the outputs apples-to-apples — only the prompt changes.

Just run the file and ALL techniques print their output in order:

        python prompt_engineering_customer_support.py

--------------------------------------------------------------------------------
 BEFORE YOU RUN
--------------------------------------------------------------------------------
 1. Install the dependencies:

        pip install langchain langchain-google-vertexai python-dotenv

 2. Create a file named ".env" in the SAME folder as this script, containing:

        GCP_PROJECT_ID=your-gcp-project-id

    (You also need Google Cloud credentials set up, e.g. via
     `gcloud auth application-default login`.)

--------------------------------------------------------------------------------
 TECHNIQUES IN THIS FILE
--------------------------------------------------------------------------------
    1. System Prompt vs User Prompt   (the foundation — read first)
    2. Zero-Shot Prompting
    3. One-Shot Prompting
    4. Few-Shot Prompting
    5. Chain of Thought (CoT)
    6. Multi-Turn Prompting
    7. Structured Output and Delimiters
    8. Prompt Templates (reusable prompts)

 Cheat-sheet reminder: always start with Role, Task, Context, Format,
 Constraints — then pick a technique from this sheet.
================================================================================
"""

# ==============================================================================
# 0. SETUP — shared config used by every example
# ==============================================================================
# This block runs once. It creates ONE `llm` object and ONE sample `TICKET`
# string that every technique reuses. Only the prompt changes between examples.

import os
import json

from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI

load_dotenv()

llm = ChatVertexAI(
    model="gemini-2.5-pro",
    project=os.getenv("GCP_PROJECT_ID"),
    location="us-central1",
    temperature=0,   # 0 = consistent, repeatable answers.
                     # Use 0 for classification / production work.
                     # Raise it (0.7-1) only for open-ended, creative tasks.
)

# The same sample ticket is reused across every technique below,
# so you can compare outputs apples-to-apples.
TICKET = (
    "Hi, I ordered a laptop stand two weeks ago and it still hasn't "
    "arrived. This is really frustrating - I need it before my trip "
    "next Monday. Can someone help?"
)

# Note: llm.invoke() accepts either a plain string (for a single combined
# prompt) or a list of role-tagged messages (for system/user/assistant turns).
# Both forms appear below.


# ==============================================================================
# 1. System Prompt vs User Prompt   — THE FOUNDATION, READ THIS FIRST
# ==============================================================================
# Two channels, two jobs.
#   system = the job description — written once, invisible to the end user,
#            applied to every request. Put your RULES here.
#   user   = today's actual request — the ticket, the log, the question.
#            Put your DATA here.
#
# Expected: {"category": "Shipping/Delivery", "priority": "P2",
#            "summary": "...", "next_action": "..."}
# The rule (reply in JSON) lives in `system` and never has to be repeated
# per ticket.

print("\n" + "=" * 78)
print(" 1. System Prompt vs User Prompt")
print("=" * 78)

messages = [
    {"role": "system", "content": (
        "You are a customer support triage analyst. Reply in JSON."
    )},
    {"role": "user", "content": TICKET},
]
response = llm.invoke(messages)
print(response.content)


# ==============================================================================
# 2. Zero-Shot Prompting
# ==============================================================================
# USE WHEN: the task is common, you need something working in seconds,
#           and you have no examples yet.
#
# Instructions only, no examples. The model works from the brief alone.
#
# Typical output: "This looks like a shipping/delivery issue. Priority: Medium.
# The customer should receive a delivery status update and an apology for the
# delay." Right idea, wrong shape — "Medium" is not one of our P1-P4 priorities,
# and it's prose, not data. This is exactly what One-Shot fixes next.

print("\n" + "=" * 78)
print(" 2. Zero-Shot Prompting")
print("=" * 78)

prompt = (
    "Classify this support ticket. "
    "Give a category and a priority.\n\n"
    + TICKET
)
response = llm.invoke(prompt)
print(response.content)


# ==============================================================================
# 3. One-Shot Prompting
# ==============================================================================
# USE WHEN: zero-shot has the right idea but the wrong shape.
#
# Add ONE worked example so the model can see the exact output shape you want.
#
# Expected: "Shipping/Delivery, P2" — short, and already in our P-scale.
# One tiny example did what a paragraph of instructions could not.

print("\n" + "=" * 78)
print(" 3. One-Shot Prompting")
print("=" * 78)

prompt = """Classify the support ticket.

Example:
Ticket: My order arrived with the wrong color, I need the right one.
Answer: Product Issue, P3

Ticket: """ + TICKET + """
Answer:"""
response = llm.invoke(prompt)
print(response.content)


# ==============================================================================
# 4. Few-Shot Prompting
# ==============================================================================
# USE WHEN: house-specific categories, thresholds, or a recurring edge case.
#
# Three to five examples teach the model your HOUSE RULES — things no single
# instruction could convey as cleanly.
#
# Expected: "Shipping/Delivery, P2" — it matches our house rule for delayed
# orders directly, and the model holds it at P2 rather than P1 because only one
# customer is affected and no money is in dispute, purely from the pattern shown
# in the examples.

print("\n" + "=" * 78)
print(" 4. Few-Shot Prompting")
print("=" * 78)

prompt = """Classify support tickets using our house rules below.

Order delayed past delivery window -> Shipping/Delivery, P2
Charged twice for one order -> Billing, P1
Request to update shipping address -> Account, P4
Item arrived damaged, customer upset -> Product Issue, P2

Ticket: """ + TICKET + """
->"""
response = llm.invoke(prompt)
print(response.content)


# ==============================================================================
# 5. Chain of Thought (CoT)
# ==============================================================================
# USE WHEN: root cause analysis, incident correlation, times / numbers /
#           multi-step logic.
#
# Ask the model to show its working before concluding. Costs more tokens —
# worth it whenever getting the reasoning right matters more than getting an
# answer fast.
#
# NOTE: This example intentionally uses a SEPARATE scenario (INCIDENT), not
# TICKET — CoT needs a multi-step reasoning problem, and our one-line ticket is
# a single, direct lookup rather than a correlation puzzle.
#
# Expected: "A: 13:20, 45 min before. B: 13:58, 7 min before. Timeouts began
# 14:05. B is the closer match. Cause: config change B." Asking "Which change
# caused the outage?" directly tends to grab the bigger, more obvious deploy (A)
# and gets it wrong.

print("\n" + "=" * 78)
print(" 5. Chain of Thought (CoT)")
print("=" * 78)

INCIDENT = (
    "Payment API timeouts began 14:05. Deploy A went live 13:20. "
    "Config change B went live 13:58. Which caused it?"
)
prompt = (
    "Think step by step: list each change with its time, "
    "compare each one to when the timeouts began, then state "
    "your conclusion clearly.\n\n" + INCIDENT
)
response = llm.invoke(prompt)
print(response.content)


# ==============================================================================
# 6. Multi-Turn Prompting
# ==============================================================================
# USE WHEN: follow-up work (classify, then draft, then escalate), or any chat
#           interface.
#
# A conversation with memory. Each turn carries every previous turn along, so
# you never repeat context the model already has.
#
# Notice the second question never mentions "laptop stand" or "Monday" — it
# doesn't have to, the whole conversation list is resent each call. Watch out:
# every turn re-sends the full history, so long chats cost more tokens.

print("\n" + "=" * 78)
print(" 6. Multi-Turn Prompting")
print("=" * 78)

conversation = [
    {"role": "system", "content": "You are a customer support triage analyst."},
    {"role": "user", "content": TICKET},
]
turn_1 = llm.invoke(conversation).content
print("Turn 1:", turn_1)

# Carry the conversation forward: append what the model said, then ask again
conversation.append({"role": "assistant", "content": turn_1})
conversation.append({"role": "user", "content": "Now draft a reply to the user."})
turn_2 = llm.invoke(conversation).content
print("Turn 2:", turn_2)


# ==============================================================================
# 7. Structured Output and Delimiters
# ==============================================================================
# USE WHEN: another system (script, spreadsheet, ticket tool) will read the
#           answer.
#
# Delimiters (###) stop messy ticket text from being read as instructions.
# "Return ONLY JSON" kills the chatty wrapper text. This is the bridge from a
# demo to something you can actually run in production.
#
# Expected: {"category":"Shipping/Delivery","priority":"P2","summary":"Order
# placed two weeks ago hasn't arrived; customer needs it before a trip next
# Monday.","next_action":"Send tracking update and expedite if possible"} —
# no greeting, no apology, straight into a spreadsheet or ticket tool.

print("\n" + "=" * 78)
print(" 7. Structured Output and Delimiters")
print("=" * 78)

RULES = """You are a customer support triage analyst.
Classify the ticket between the ### markers.
Return ONLY valid JSON with keys: category, priority, summary, next_action.
If the ticket is too vague to classify, set every field to "UNKNOWN"."""

messages = [
    {"role": "system", "content": RULES},
    {"role": "user", "content": "###\n" + TICKET + "\n###"},
]
response = llm.invoke(messages)
print(response.content)

# Parse straight into a dict for downstream systems.
# Real-world gotcha: even after "Return ONLY JSON", models often wrap the
# answer in a ```json ... ``` markdown fence. json.loads() can't read those
# backticks, so we strip the fence first before parsing.
text = response.content.strip()
if text.startswith("```"):
    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
data = json.loads(text)
print(data["priority"])


# ==============================================================================
# 8. Prompt Templates (reusable prompts)
# ==============================================================================
# HOW PROMPTS ACTUALLY LIVE IN PRODUCTION CODE
#
# A prompt template is a reusable prompt with blanks — write the wording once,
# and only the variables change per request.
#
# This is what sits behind a real triage app: the template is written and
# reviewed once, then called thousands of times a day with a different ticket
# dropped into the blank.

print("\n" + "=" * 78)
print(" 8. Prompt Templates (reusable prompts)")
print("=" * 78)

from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["ticket", "categories"],
    template=(
        "Classify the ticket into one of: {categories}.\n\n"
        "Ticket: {ticket}\n"
        "Category:"
    ),
)

prompt = template.format(
    ticket=TICKET,
    categories="Shipping/Delivery, Billing, Product Issue, Account",
)
response = llm.invoke(prompt)
print(response.content)