"""
chatbot.py
Ties together intent extraction, retrieval, and LLM reasoning to produce
the assistant's actual response: a follow-up question, a grounded
recommendation, or a product comparison.
"""

import os
import time
import json
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError, ClientError

from intent_extraction import extract_intent
from retrieval import ProductRetriever

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Use the stronger model first here — reasoning/explanation quality matters
# more than raw speed for this step.
MODEL_CANDIDATES = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]


def call_llm(prompt, max_retries=3):
    for model_name in MODEL_CANDIDATES:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return response.text.strip()
            except ServerError:
                wait = 2 ** attempt
                print(f"{model_name} overloaded (503), retrying in {wait}s...")
                time.sleep(wait)
            except ClientError as e:
                if getattr(e, "status_code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"{model_name} rate-limited (429), switching model...")
                    break
                else:
                    raise
        print(f"{model_name} exhausted, falling back to next model...")
    raise RuntimeError("All Gemini models failed after retries")


def ask_followup(intent):
    slot_prompts = {
        "budget": "What's your budget range?",
        "category": "What kind of product are you looking for — clothing, footwear, a phone, or a laptop?",
        "gender": "Is this for a men's or women's item?",
        "use_case": "What will you mainly be using it for?",
    }
    questions = [slot_prompts.get(slot, f"Could you clarify your {slot}?")
                 for slot in intent["missing_slots"]]
    return " ".join(questions)


def format_candidates_for_prompt(candidates_df):
    lines = []
    for _, row in candidates_df.iterrows():
        specs = json.loads(row["specs"]) if isinstance(row["specs"], str) else row["specs"]
        lines.append(
            f"- {row['title']} | Brand: {row['brand']} | Price: ₹{row['price']} | "
            f"Specs: {json.dumps(specs, default=str)}"
        )
    return "\n".join(lines)


def generate_recommendation(user_message, intent, candidates_df):
    if candidates_df.empty:
        return "I couldn't find any matching products in the catalog for that request. Could you widen your budget or try a different category?"

    candidate_text = format_candidates_for_prompt(candidates_df)

    prompt = f"""You are a helpful, knowledgeable shopping assistant (virtual sales associate).

User asked: "{user_message}"
Understood use case: {intent.get('use_case')}
Budget max: {intent.get('budget_max')}

Here are candidate products retrieved from the catalog:
{candidate_text}

Task:
1. Select the 3 best-fitting products from the list above for the user's actual need
   (consider specs, not just price/title — e.g. camera quality for photography,
   RAM/CPU for engineering work, color/season fit for clothing).
2. For each, give a short (1-2 sentence) explanation of WHY it fits their need.
3. If none of the candidates are a strong fit, say so honestly rather than forcing a pick.
4. Keep the tone natural and conversational, like a helpful salesperson — not a spec sheet dump.
5. Do not recommend any product not in the candidate list above.

Respond in plain text, not JSON.
"""
    return call_llm(prompt)


def generate_comparison(user_message, intent, candidates_df):
    targets = intent.get("comparison_targets", [])

    if not candidates_df.empty:
        candidate_text = format_candidates_for_prompt(candidates_df)
        grounding_note = f"Here is what I found in the catalog for these products:\n{candidate_text}\n"
    else:
        grounding_note = (
            "Note: these specific products are not in my local catalog, so I'll compare them "
            "using general knowledge instead of catalog data.\n"
        )

    prompt = f"""You are a helpful shopping assistant.

User asked: "{user_message}"
They want a comparison between: {', '.join(targets)}

{grounding_note}

Task:
Provide a structured comparison covering: Camera, Performance, Battery, Display,
Software/OS, Pricing, and a short Pros/Cons summary for each product.
If you are using general knowledge rather than catalog data, keep specs at a
general, well-known level rather than inventing precise numbers.
Keep it clear and easy to scan.
"""
    return call_llm(prompt)


def generate_chitchat_reply(user_message):
    prompt = f"""You are a friendly shopping assistant chatbot for an e-commerce site.
The user said: "{user_message}"
This isn't a shopping request. Reply briefly and warmly, and invite them to ask
about products you can help with (phones, laptops, clothing, footwear)."""
    return call_llm(prompt)


def handle_message(user_message, retriever):
    intent = extract_intent(user_message)

    if intent is None:
        return "Sorry, I had trouble understanding that — could you rephrase?"

    intent_type = intent.get("intent_type")

    if intent_type == "chitchat":
        return generate_chitchat_reply(user_message)

    if intent.get("missing_slots"):
        return ask_followup(intent)

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


if __name__ == "__main__":
    retriever = ProductRetriever()

    test_queries = [
        "Suggest some black shirts for my upcoming Goa trip",
        "Recommend the best camera phone under 40000",
        "Compare iPhone 16 Pro and Samsung S25 Ultra",
        "Help me buy a laptop for Computer Engineering studies, budget 70000",
    ]

    for q in test_queries:
        print(f"\n{'='*80}\nUser: {q}\n{'-'*80}")
        reply = handle_message(q, retriever)
        print(f"Bot: {reply}")
        time.sleep(15)  # stay well under free-tier rate limits across two LLM calls per query