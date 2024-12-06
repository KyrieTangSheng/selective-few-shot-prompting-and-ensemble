from typing import List, Dict, Optional, Tuple
from .template_manager import TemplateManager, PromptTemplate
import tiktoken
from dataclasses import dataclass

@dataclass
class PromptResult:
    prompt: str
    template_name: str
    cache_key: str
    estimated_tokens: int

class PromptBuilder:
    def __init__(
        self,
        template_manager: TemplateManager,
        max_tokens: int = 4000
    ):
        self.template_manager = template_manager
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def _format_examples(
        self,
        template: PromptTemplate,
        examples: List[Dict],
        max_examples: Optional[int] = None
    ) -> str:
        max_examples = max_examples or template.max_examples
        formatted_examples = []
        
        for i, example in enumerate(examples[:max_examples]):
            example_dict = {
                'index': i+1,
                'text': example['text'],
                'sentiment': example['sentiment'],
                'sentiment_label': 'positive' if example['sentiment'] == 1 else 'negative'
            }
            
            try:
                formatted_example = template.example_format.format(**example_dict)
                formatted_examples.append(formatted_example)
            except KeyError as e:
                minimal_dict = {
                    'index': i+1,
                    'text': example['text'],
                    'sentiment': example['sentiment']
                }
                formatted_example = template.example_format.format(**minimal_dict)
                formatted_examples.append(formatted_example)
                
        return template.separator.join(formatted_examples)
    
    def build_prompt(
        self,
        examples: List[Dict],
        input_text: str,
        template_name: str = 'basic'
    ) -> PromptResult:
        """
        Build prompt with caching and token management.
        
        Args:
            examples: List of example dictionaries
            input_text: Input text to classify
            template_name: Name of template to use
            
        Returns:
            PromptResult containing prompt and metadata
        """
        template = self.template_manager.get_template(template_name)
        cache_key = self.template_manager._compute_cache_key(
            template_name, examples, input_text
        )
        
        if cache_key in self.template_manager.cache:
            return self.template_manager.cache[cache_key]
        
        formatted_examples = self._format_examples(template, examples)
        
        prompt = template.template.format(
            instruction=template.instruction,
            examples=formatted_examples,
            input_text=input_text
        )
        
        estimated_tokens = self._count_tokens(prompt)
        
        if estimated_tokens > self.max_tokens:
            left, right = 1, len(examples)
            while left < right:
                mid = (left + right + 1) // 2
                formatted_examples = self._format_examples(template, examples, mid)
                prompt = template.template.format(
                    instruction=template.instruction,
                    examples=formatted_examples,
                    input_text=input_text
                )
                if self._count_tokens(prompt) <= self.max_tokens:
                    left = mid
                else:
                    right = mid - 1
                    
            formatted_examples = self._format_examples(template, examples, left)
            prompt = template.template.format(
                instruction=template.instruction,
                examples=formatted_examples,
                input_text=input_text
            )
            estimated_tokens = self._count_tokens(prompt)
        
        result = PromptResult(
            prompt=prompt,
            template_name=template_name,
            cache_key=cache_key,
            estimated_tokens=estimated_tokens
        )
        
        # Cache result
        self.template_manager.cache[cache_key] = result
        self.template_manager._save_cache()
        
        return result