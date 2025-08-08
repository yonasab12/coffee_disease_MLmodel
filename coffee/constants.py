# coffee/constants.py
DISEASE_LABELS = ['Healthy', 'Coffee Rust', 'Leaf Miner', 'Phoma', 'Cercospora']

OPENAI_PROMPT_TEMPLATE = """
As a coffee plant pathologist, provide detailed information about {disease} infection.
Current infection level: {infection_level} (Confidence: {confidence}%).

Structure your response EXACTLY like this:

Summary: [1-2 sentence disease overview]
Organic Treatment: [3-4 organic solutions]
Chemical Treatment: [2-3 chemical products with active ingredients]
Environmental Advice: [3-4 prevention strategies]

Focus specifically on {infection_level_lower} level infections.
Provide actionable recommendations suitable for coffee farmers.
"""