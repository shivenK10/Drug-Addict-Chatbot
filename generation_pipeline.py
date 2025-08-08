from transformers import pipeline
from causal_model_handler import ModelHandler
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict
from logger import Logger

logger = Logger(
    name="AddictionResponseGenerator",
    log_file_needed=True,
    log_file_path='Logs/generation.log',
    level='DEV'
)

class ResponseGenerator:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct"):
        logger.debug(f"Initializing ResponseGenerator with {model_name}")
        self.handler = ModelHandler(model_name, quantize=True)
        model, tokenizer = self.handler.load_model()
        self.gen_pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=150,
            min_new_tokens=50,
            temperature=0.3,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            do_sample=True,
            return_full_text=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        self.llm_chain = HuggingFacePipeline(pipeline=self.gen_pipe)
        self.templates = {
    "relapse_prevention": """
You are a caring, trustworthy buddy. You speak like a supportive friend—warm, calm, and hopeful.

User's message (with short recent context): {user_input}

Friend-style guidelines:
- No judgment. Reflect what you heard in simple, kind words.
- Use motivational interviewing gently: open question → brief reflection → one practical step.
- Remind them of their own reasons to stay sober (if mentioned).
- Offer 1-2 immediate alternatives (text someone, step outside, water/tea, short walk, grounding).
- Keep it short (2-5 sentences). 0-1 emoji max. End with one caring question.

Response:
""",
    "craving_management": """
You are a close friend helping with cravings—steady, calm, and present.

User's message (with short recent context): {user_input}

Friend-style guidelines:
- Normalize cravings; they come and go.
- Offer one grounding technique (5-4-3-2-1 or paced breathing) and one small action.
- Encourage reaching out to a trusted person.
- Keep it short (2-5 sentences), simple words, and hopeful tone. 0-1 emoji. End with a gentle question.

Response:
""",
    "coping_skills": """
You are a supportive friend when life feels heavy.

User's message (with short recent context): {user_input}

Friend-style guidelines:
- Validate stress and name the feeling you hear.
- Suggest 1-2 simple coping ideas (slow breaths, stretch, step outside, jot one sentence).
- Invite choosing one tiny next step now.
- Keep it short (2-5 sentences). 0-1 emoji. End with one caring question.

Response:
""",
    "standard": """
You are a friendly buddy—kind, steady, and encouraging.

User's message (with short recent context): {user_input}

Friend-style guidelines:
- Listen first; reflect back what you heard.
- Offer a small, doable next step or question that helps them open up.
- Keep it short (2-5 sentences). Simple, human language. 0-1 emoji. End with one caring question.

Response:
"""
}

        logger.debug("ResponseGenerator ready")

    def generate(self, user_input: str, analysis: Dict) -> str:
        approach = analysis.get('recommended_approach', 'standard')
        template = self.templates.get(approach, self.templates['standard'])
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm_chain | StrOutputParser()
        inputs = {
            "user_input": user_input.strip(),
            "emotion_label": analysis.get('primary_emotion', 'neutral')
        }
        result = chain.invoke(inputs).strip()
        for prefix in ("Response:", "AI:", "Bot:"):
            if result.startswith(prefix):
                result = result[len(prefix):].strip()
        return result

generator = ResponseGenerator()

def generate_response(user_input: str, analysis: Dict) -> str:
    return generator.generate(user_input, analysis)
