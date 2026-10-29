import os
import sys
import json
import urllib.request

DEFAULT_MISTRAL_API_KEY = "mstrl_7BhNSLycMwmsCzGalbsbajlEIR8K5lyI_3FyT9d"

SYSTEM_PROMPT = """You are System 2: The Reasoning Engine for a 4-DOF Dobot Magician edge robot arm.
Your job is to parse human natural language instructions (including ambiguous, casual, or complex requests) into strict, structured JSON target specifications for the edge robot controller.

Available target roles:
1. "pick": the object to grasp/pick up.
2. "place": the target container, bin, platform, or drop zone.
3. "avoid": any obstacles, hazards, or items to stay clear of or arc over.

Always identify the key visual attributes (color, shape/type):
Supported colors: red, green, blue, yellow, purple, cyan, orange.
Supported object types: wrench, cube, block, cup, bin, platform, zone, cylinder, obstacle, bar, tray.

Response format:
Respond ONLY with a valid JSON object matching this schema, with no markdown code fences and no conversational filler:
{
  "thought": "brief 1-sentence reasoning",
  "pick": {"color": "<color>", "type": "<type>"},
  "place": {"color": "<color>", "type": "<type>"},
  "avoid": [{"color": "<color>", "type": "<type>"}]
}

Few-shot examples:

Example 1:
User: "Hey robot, pick up the red wrench but watch out for that green cup, and drop it in the blue bin."
Assistant:
{
  "thought": "User wants to pick red wrench, place into blue bin, and avoid green cup.",
  "pick": {"color": "red", "type": "wrench"},
  "place": {"color": "blue", "type": "bin"},
  "avoid": [{"color": "green", "type": "cup"}]
}

Example 2:
User: "Move the yellow cube onto the cyan platform. Make sure you don't hit the red block standing in between!"
Assistant:
{
  "thought": "Pick target is yellow cube, destination is cyan platform, obstacle is red block.",
  "pick": {"color": "yellow", "type": "cube"},
  "place": {"color": "cyan", "type": "platform"},
  "avoid": [{"color": "red", "type": "block"}]
}

Example 3:
User: "Clear the purple bar over to the orange tray. Avoid the red obstacle in the middle."
Assistant:
{
  "thought": "Purple bar is pick target, orange tray is destination, red obstacle must be avoided.",
  "pick": {"color": "purple", "type": "bar"},
  "place": {"color": "orange", "type": "tray"},
  "avoid": [{"color": "red", "type": "obstacle"}]
}

Example 4:
User: "Can you grab that elongated green widget and deposit it inside the blue target box? Steer well clear of the red cylinder."
Assistant:
{
  "thought": "Target object is green widget, place is blue box, obstacle is red cylinder.",
  "pick": {"color": "green", "type": "wrench"},
  "place": {"color": "blue", "type": "box"},
  "avoid": [{"color": "red", "type": "cylinder"}]
}
"""

class System2ReasoningEngine:
    """
    System 2: The Reasoning Engine (Pure Mistral VLM / LLM).
    Zero regex. Zero heuristic rule parsers.
    Strictly queries Mistral LLM with few-shot guidance and receives JSON targets.
    """
    def __init__(self, api_key=None, model="open-mistral-7b"):
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY", DEFAULT_MISTRAL_API_KEY)
        self.model = model
        self.system_prompt = SYSTEM_PROMPT

    def parse_instruction(self, instruction: str) -> dict:
        """
        Directly queries Mistral API with the human instruction and few-shot system prompt.
        Parses JSON output directly from the LLM.
        """
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": instruction}
            ],
            "temperature": 0.0
        }

        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw_content = data["choices"][0]["message"]["content"].strip()
            
            # Remove any markdown code fences if output by LLM
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.startswith("```"):
                raw_content = raw_content[3:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
            raw_content = raw_content.strip()

            parsed = json.loads(raw_content)
            return parsed

if __name__ == "__main__":
    engine = System2ReasoningEngine()
    test_prompts = [
        "Pick up the elongated green wrench and drop it in the blue tray, avoiding the red obstacle",
        "Can you grab that yellow block and place it on the cyan platform, don't hit the red cup!",
        "Transfer purple bar to orange bin",
        "Clear the green wrench over to the blue destination, watch out for the red cylinder in the way!"
    ]
    for p in test_prompts:
        print(f"\nUser: {p}")
        out = engine.parse_instruction(p)
        print("Mistral LLM Parsed Output:")
        print(json.dumps(out, indent=2))
