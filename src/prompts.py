"""
Trait prompt pairs and evaluation questions loader.
Loads all traits from src/data/prompts.json (single source of truth).
"""
import json
from pathlib import Path

def load_prompts_json():
    """Load all traits from prompts.json."""
    prompts_file = Path(__file__).parent / "config" / "prompts.json"
    if prompts_file.exists():
        try:
            with open(prompts_file, 'r') as f:
                data = json.load(f)
                return data.get('benchmark_prompts', {}).get('traits', {})
        except Exception as e:
            print(f"Warning: Could not load prompts.json: {e}")
    return {}

def load_custom_traits():
    """Load custom traits from the cache file."""
    custom_traits = {}
    custom_traits_file = Path(__file__).parent / "data" / "custom_traits.json"
    if custom_traits_file.exists():
        try:
            with open(custom_traits_file, 'r') as f:
                data = json.load(f)
                for trait_name, trait_data in data.items():
                    custom_traits[trait_data["id"]] = trait_data["prompt_pairs"]
        except:
            pass
    return custom_traits

def load_custom_questions():
    """Load custom trait evaluation questions."""
    custom_questions = {}
    custom_traits_file = Path(__file__).parent / "data" / "custom_traits.json"
    if custom_traits_file.exists():
        try:
            with open(custom_traits_file, 'r') as f:
                data = json.load(f)
                for trait_name, trait_data in data.items():
                    custom_questions[trait_data["id"]] = trait_data["evaluation_questions"]
        except:
            pass
    return custom_questions

def load_custom_eval_prompts():
    """Load custom trait evaluation prompts (Chen et al. methodology)."""
    custom_eval_prompts = {}
    custom_traits_file = Path(__file__).parent / "data" / "custom_traits.json"
    if custom_traits_file.exists():
        try:
            with open(custom_traits_file, 'r') as f:
                data = json.load(f)
                for trait_name, trait_data in data.items():
                    # Get eval_prompt if it exists (new Chen et al. format)
                    eval_prompt = trait_data.get("eval_prompt", None)
                    if eval_prompt:
                        custom_eval_prompts[trait_data["id"]] = eval_prompt
        except:
            pass
    return custom_eval_prompts

# Load all traits from prompts.json (single source of truth)
PROMPTS_JSON_TRAITS = load_prompts_json()

# Build prompt pairs dictionary from JSON
PROMPT_PAIRS = {}
for trait_id, trait_data in PROMPTS_JSON_TRAITS.items():
    if 'prompt_pairs' in trait_data:
        PROMPT_PAIRS[trait_id] = trait_data['prompt_pairs']

# Build evaluation questions dictionary from JSON
EVALUATION_QUESTIONS = {}
for trait_id, trait_data in PROMPTS_JSON_TRAITS.items():
    if 'eval_questions' in trait_data:
        EVALUATION_QUESTIONS[trait_id] = trait_data['eval_questions']

def get_prompt_pairs(trait_id):
    """Get prompt pairs for a specific trait (built-in or custom)."""
    # First check JSON traits
    if trait_id in PROMPT_PAIRS:
        return PROMPT_PAIRS[trait_id]
    
    # Then check custom traits
    custom_traits = load_custom_traits()
    return custom_traits.get(trait_id, [])

def get_evaluation_questions(trait_id):
    """Get evaluation questions for a specific trait (built-in or custom)."""
    # First check JSON questions
    if trait_id in EVALUATION_QUESTIONS:
        return EVALUATION_QUESTIONS[trait_id]
    
    # Then check custom questions
    custom_questions = load_custom_questions()
