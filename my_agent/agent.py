"""ADK root agent for identity-document fraud detection.

Wraps the trained FREUID model behind a single tool
``classify_document`` so the LLM can call it with an image path and
receive a calibrated fraud score + verdict.

Run interactively (after `pip install google-adk`):

    # from repo root with GOOGLE_API_KEY exported
    adk run my_agent            # or: adk web my_agent

Run a one-shot query:

    adk run my_agent --query "classify dataset/train/000001.jpg"
"""

from __future__ import annotations

from google.adk.agents.llm_agent import Agent

from my_agent.classifier import classify_document


# Tool exposed to the LLM. Docstring becomes the tool description in ADK.
def classify_document_tool(image_path: str) -> dict:
    """Classify an identity-document image.

    Args:
        image_path: path to a JPG/PNG image of an identity document
            (e.g. "dataset/train/000001.jpg" or an absolute path).

    Returns:
        dict with keys: image_path, fraud_score (0..1, higher = more
        likely fraudulent), verdict ("bona_fide" | "review" |
        "fraudulent"), confidence (0..1), and model backbone name.
    """
    return classify_document(image_path)


root_agent = Agent(
    model="gemini-2.5-flash",
    name="freuid_fraud_agent",
    description=(
        "Identity-document fraud-detection agent. Given the path to an "
        "ID document image, returns a calibrated fraud score and verdict "
        "using a model trained on the FREUID dataset."
    ),
    instruction=(
        "You are a fraud-detection assistant for identity documents "
        "(passports, driver's licences, national ID cards). When the user "
        "gives you an image path, call the `classify_document_tool` once "
        "and report the result in plain language. If the verdict is "
        "'fraudulent', warn the user and recommend rejecting the document. "
        "If 'review', recommend secondary human inspection. If "
        "'bona_fide', confirm it appears genuine but note the model is a "
        "first-pass filter only. Always include the numeric fraud_score "
        "and confidence in your reply."
    ),
    tools=[classify_document_tool],
)