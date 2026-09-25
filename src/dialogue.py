"""
dialogue.py
Manages multi-turn conversation state so follow-up answers (e.g. answering
"what category?" with just "shoes") get merged into the original intent
instead of being treated as a brand-new, context-free query.
"""

import json
from chatbot import (
    call_llm, ask_followup, generate_recommendation,
    generate_comparison, generate_chitchat_reply
)
from intent_extraction import extract_intent
from retrieval import ProductRetriever


class ConversationState:
    """Holds the pending intent across turns while slots are being filled."""

    def __init__(self):
        self.pending_intent = None   # intent dict awaiting missing_slots
        self.original_message = None
        self.history = []            # list of (role, text) for optional context

    def reset(self):
        self.pending_intent = None
        self.original_message = None

    def has_pending(self):
        return self.pending_intent is not None


def merge_followup_answer(pending_intent, followup_message):
    """
    Uses the LLM to merge a short follow-up reply (e.g. "shoes", "under 2000",
    "men's") into the previously extracted intent, filling missing_slots.
    """
    prompt = f"""You previously extracted this shopping intent from a user, but some
information was missing:

{json.dumps(pending_intent, indent=2)}

The user has now replied to your follow-up question with: "{followup_message}"

Update the intent JSON by filling in whatever the reply answers. Keep all
previously known fields unless the reply changes them. Recompute missing_slots
based on what's now known (empty list if nothing important remains missing).

IMPORTANT: "category" must be exactly one of these four values only:
"smartphone", "laptop", "clothing", "footwear".
Map synonyms accordingly — e.g. "shoes"/"sneakers"/"sandals" -> "footwear",
"shirt"/"t-shirt"/"top" -> "clothing", "phone"/"mobile" -> "smartphone".

Return ONLY the updated JSON, same schema as before, no markdown fences.
"""
    raw = call_llm(prompt)
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json\n", "", 1).replace("json", "", 1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return pending_intent  # fall back to unmerged intent rather than crashing


def route_intent(user_message, intent, retriever):
    """Given a *complete* intent (no missing_slots), produce the final reply."""
    intent_type = intent.get("intent_type")

    if intent_type == "chitchat":
        return generate_chitchat_reply(user_message)

    if intent_type == "comparison":
        candidates = retriever.search(
            query=" ".join(intent.get("comparison_targets", [])),
            top_k=6,
        )
        return generate_comparison(user_message, intent, candidates)

    if intent_type == "recommendation":
        query_text = intent.get("use_case") or user_message
        candidates = retriever.search(
            query=query_text,
            category=intent.get("category"),
            max_price=intent.get("budget_max"),
            min_price=intent.get("budget_min"),
            top_k=10,
        )
        return generate_recommendation(user_message, intent, candidates)

    return "I'm not sure how to help with that yet — try asking about a product recommendation or comparison."


def handle_turn(user_message, state: ConversationState, retriever: ProductRetriever):
    """
    Main entry point for one user message. Handles both fresh queries and
    follow-up answers to a pending clarification question.
    """
    state.history.append(("user", user_message))

    if state.has_pending():
        # This message is answering a previous follow-up question
        updated_intent = merge_followup_answer(state.pending_intent, user_message)
        
        if updated_intent.get("missing_slots"):
            state.pending_intent = updated_intent
            reply = ask_followup(updated_intent)
        else:
            combined_message = f"{state.original_message} ({user_message})"
            reply = route_intent(combined_message, updated_intent, retriever)
            state.reset()

        state.history.append(("bot", reply))
        return reply

    # Fresh message — extract intent as usual
    intent = extract_intent(user_message)

    if intent is None:
        reply = "Sorry, I had trouble understanding that — could you rephrase?"
        state.history.append(("bot", reply))
        return reply

    if intent.get("missing_slots"):
        state.pending_intent = intent
        state.original_message = user_message
        reply = ask_followup(intent)
    else:
        reply = route_intent(user_message, intent, retriever)

    state.history.append(("bot", reply))
    return reply


if __name__ == "__main__":
    retriever = ProductRetriever()
    state = ConversationState()

    # Simulated multi-turn conversation
    conversation = [
        "I am joining a gym next week. What shoes and clothes should I buy?",
        "shoes, budget around 3000",
    ]

    for msg in conversation:
        print(f"\nUser: {msg}")
        reply = handle_turn(msg, state, retriever)
        print(f"Bot: {reply}")