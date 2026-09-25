import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError, ClientError

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_CANDIDATES = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]

VALID_CATEGORIES = ["smartphone", "laptop", "clothing", "footwear"]

SYSTEM_INSTRUCTIONS = f"""You are an intent-extraction module for a shopping assistant chatbot.
Given a user's message, extract structured shopping intent as JSON.

Valid categories: {VALID_CATEGORIES}

Return ONLY valid JSON, no markdown fences, no explanation, matching this schema:
{{
  "intent_type": "recommendation" | "comparison" | "chitchat",
  "category": one of {VALID_CATEGORIES} or null if unclear/multiple,
  "budget_max": number in INR or null,
  "budget_min": number in INR or null,
  "use_case": short string describing what the product is for, or null,
  "keywords": list of relevant descriptive keywords (color, style, brand, etc.),
  "comparison_targets": list of product names if intent_type is "comparison", else [],
  "missing_slots": list of important missing info the assistant should ask about
                   (e.g. "budget", "category", "gender", "use_case"). Empty list if
                   there's enough info to give a useful recommendation.
}}

Rules:
- If the message clearly implies a category (e.g. "laptop", "shirt", "phone"), fill it in.
- If the message names two or more specific products to compare, set intent_type to "comparison"
  and list them in comparison_targets.
- If the message is just chit-chat/greeting with no shopping intent, set intent_type to "chitchat"
  and leave other fields null/empty.
- Only flag "missing_slots" for things that would meaningfully change the recommendation.
  Don't over-ask — if a use-case is given, budget being unknown is fine to proceed without asking,
  unless the category is something where budget matters a lot (electronics).
- budget_max/budget_min should be plain numbers in INR (e.g. 40000), not strings.
"""


def extract_intent(user_message, max_retries=3):
    prompt = f"{SYSTEM_INSTRUCTIONS}\n\nUser message: \"{user_message}\"\n\nJSON:"
    raw = None

    for model_name in MODEL_CANDIDATES:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                raw = response.text.strip()

                if raw.startswith("```"):
                    raw = raw.strip("`")
                    raw = raw.replace("json\n", "", 1).replace("json", "", 1)

                return json.loads(raw)

            except ServerError:
                wait = 2 ** attempt
                print(f"{model_name} overloaded (503), retrying in {wait}s...")
                time.sleep(wait)

            except ClientError as e:
                if getattr(e, "status_code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 30  # free-tier quota resets roughly every 30-60s
                    print(f"{model_name} rate-limited (429), waiting {wait}s before switching model...")
                    time.sleep(wait)
                    break  # stop retrying this model, move to next one in the outer loop
                else:
                    raise

            except json.JSONDecodeError:
                print(f"Failed to parse JSON from {model_name} output:")
                print(raw)
                return None

        print(f"{model_name} exhausted, falling back to next model...")

    raise RuntimeError("All Gemini models failed after retries")


if __name__ == "__main__":
    test_queries = [
        "Suggest some black shirts for my upcoming Goa trip",
        "Recommend the best camera phone under 40000",
        "Compare iPhone 16 Pro and Samsung S25 Ultra",
        "Help me buy a laptop for Computer Engineering studies",
        "I am joining a gym next week. What shoes and clothes should I buy?",
        "hi how are you",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        result = extract_intent(q)
        print(json.dumps(result, indent=2))
        time.sleep(13)  # stay under the 5-requests/minute free-tier limit