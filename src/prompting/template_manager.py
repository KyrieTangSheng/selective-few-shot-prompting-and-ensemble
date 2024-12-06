from typing import List, Dict, Optional
import json
import os
from dataclasses import dataclass
import hashlib
import pickle

@dataclass
class PromptTemplate:
    name: str
    template: str
    example_format: str
    instruction: str
    separator: str = "\n"
    max_examples: int = 5

class TemplateManager:
    def __init__(self, cache_dir: str = "./cache/prompts"):
        """
        Initialize template manager with caching support.
        
        Args:
            cache_dir: Directory to store prompt cache
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Default templates
        self.templates = {
            'basic': PromptTemplate(
                name='basic',
                template="""Analyze the sentiment of the review as positive (1) or negative (0).

Examples:
{examples}

Review to analyze:
{input_text}""",
                example_format="Review: {text}\nSentiment: {sentiment}\n",
                separator="\n",
                instruction="Classify sentiment as positive (1) or negative (0)",
                max_examples=3
            ),
            
            'structured': PromptTemplate(
                name='structured',
                template="""You are a sentiment analysis expert. Classify the sentiment of each review as positive (1) or negative (0).

Instructions:
{instruction}

Examples:
{examples}

Review to analyze:
{input_text}

Provide your analysis as a JSON object with 'sentiment', 'confidence', and 'reasoning' fields.""",
                example_format="Review: {text}\nSentiment: {sentiment} ({sentiment_label})\n",
                separator="\n",
                instruction="Analyze sentiment with confidence and reasoning.",
                max_examples=3
            ),
            
            'cot': PromptTemplate(
                name='cot',
                template="""You are a sentiment analysis expert. Think step by step to determine if the review is positive (1) or negative (0).

Examples:
{examples}

Review to analyze:
{input_text}

Let's approach this step by step:
1. Identify key sentiment words
2. Consider context and tone
3. Weigh positive vs negative elements
4. Make final determination

Provide your analysis as a JSON object with 'sentiment', 'confidence', and 'reasoning' fields.""",
                example_format="Review: {text}\nAnalysis:\n- Key words: ...\n- Context: ...\n- Overall: {sentiment} ({sentiment_label})\n",
                separator="\n\n",
                instruction="Analyze sentiment using step-by-step reasoning.",
                max_examples=2
            )
        }
        self._register_default_templates()
        self.cache = {}
        self._load_cache()
       
    
    def _load_cache(self):
        """Load prompt cache from disk."""
        cache_file = os.path.join(self.cache_dir, "prompt_cache.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.cache = pickle.load(f)
    
    def _save_cache(self):
        """Save prompt cache to disk."""
        cache_file = os.path.join(self.cache_dir, "prompt_cache.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def _compute_cache_key(self, template_name: str, examples: List[Dict], input_text: str) -> str:
        """Compute cache key for a prompt configuration."""
        key_data = {
            'template': template_name,
            'examples': [str(ex) for ex in examples],
            'input': input_text
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def get_template(self, name: str) -> PromptTemplate:
        """Get a prompt template by name."""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        return self.templates[name]
    
    def add_template(self, template: PromptTemplate):
        """Add a new template."""
        self.templates[template.name] = template
    
    def _register_default_templates(self):
        # Zero-shot template
        self.templates['zeroshot'] = PromptTemplate(
            name='zeroshot',
            instruction="Analyze the sentiment of movie reviews.",
            template="""You are a sentiment analyzer. Analyze the sentiment of the following movie review and classify it as either positive (1) or negative (0). Provide your prediction and confidence score between 0 and 1.

Review: {input_text}

Output your response in JSON format:
{{
    "sentiment": (0 or 1),
    "confidence": (float between 0 and 1),
    "reasoning": "brief explanation"
}}""",
            example_format="",  # No example formatting needed for zero-shot
            separator="\n",
            max_examples=0
        )
        
        # Few-shot template
        self.templates['fewshot'] = PromptTemplate(
            name='fewshot',
            instruction="Learn from examples to analyze movie review sentiment.",
            template="""You are a sentiment analyzer. Here are some examples of movie review sentiment analysis:

{examples}

Now analyze the sentiment of this new review and classify it as either positive (1) or negative (0). Provide your prediction and confidence score between 0 and 1.

Review: {input_text}

Output your response in JSON format:
{{
    "sentiment": (0 or 1),
    "confidence": (float between 0 and 1),
    "reasoning": "brief explanation"
}}""",
            example_format="Review: {text}\nSentiment: {sentiment}\n",
            separator="\n\n",
            max_examples=3
        )
