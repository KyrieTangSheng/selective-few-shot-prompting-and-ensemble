from openai import OpenAI
from typing import Optional, Dict, List, Any
import json
import os
import pickle
import hashlib
from datetime import datetime
import logging
from .config import LLMConfig
from prompting.prompt_builder import PromptResult
from .rate_limiter import RateLimiter

class LLMInterface:
    def __init__(
        self,
        api_key: str,
        config: Optional[LLMConfig] = None,
        cache_responses: bool = True
    ):
        self.client = OpenAI(api_key=api_key)
        self.config = config or LLMConfig()
        self.cache_responses = cache_responses
        
        if cache_responses:
            os.makedirs(self.config.cache_dir, exist_ok=True)
            self.cache_file = os.path.join(self.config.cache_dir, "response_cache.pkl")
            self.cache = self._load_cache()
            
        self.setup_logging()
        
        self.usage_stats = {
            'total_tokens': 0,
            'total_cost': 0,
            'total_requests': 0,
            'cache_hits': 0,
            'api_calls': 0
        }
        
        self.rate_limiter = RateLimiter(
            requests_per_minute=40,  
            tokens_per_minute=80000, 
            min_delay=0.5 
        )
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("llm_interface.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_cache(self) -> Dict:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def _save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def _compute_cache_key(self, prompt: str, config: Dict[str, Any]) -> str:
        key_data = {
            'prompt': prompt,
            'config': config
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def _update_usage_stats(self, response):
        try:
            if hasattr(response, 'usage'):
                self.total_tokens += response.usage.total_tokens
                self.total_cost += (response.usage.total_tokens * 0.002 / 1000)
        except Exception as e:
            # If we can't update stats, just continue without failing
            pass
    
    def _make_request(self, prompt: str) -> Any:
        def _request():
            estimated_tokens = len(prompt.split()) * 1.3
            
            self.rate_limiter.wait_if_needed(int(estimated_tokens))
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                timeout=self.config.timeout
            )
            
            if hasattr(response, 'usage'):
                total_tokens = response.usage.total_tokens
                self.rate_limiter.wait_if_needed(total_tokens)
            
            return response
        
        return self.retry_strategy.execute(_request)
    
    def _parse_response(self, response: Any) -> int:
        try:
            content = response.choices[0].message.content.strip().lower()
            
            import re
            numbers = re.findall(r'\d+', content)
            
            if not numbers:
                if any(word in content for word in ['negative', 'neg']):
                    return 0
                elif any(word in content for word in ['positive', 'pos']):
                    return 1
                raise ValueError(f"No sentiment value found in response: {content}")
            
            sentiment = int(numbers[-1])
            
            if sentiment not in [0, 1]:
                raise ValueError(f"Invalid sentiment value: {sentiment}")
            
            return sentiment
            
        except Exception as e:
            self.logger.error(f"Error parsing response: {str(e)}")
            self.logger.debug(f"Raw response content: {content}")
            raise
    
    def predict(self, prompt_result: PromptResult) -> Dict:
        """Make a single prediction."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis expert. Output your response as a JSON object with 'sentiment' (0 or 1), 'confidence' (0-1), and 'reasoning' fields."},
                    {"role": "user", "content": prompt_result.prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            content = response.choices[0].message.content
            
            try:
                result = json.loads(content)
                if isinstance(result, dict) and 'sentiment' in result:
                    result['sentiment'] = int(result['sentiment'])
                    return result
            except (json.JSONDecodeError, ValueError):
                pass
                
            sentiment = 1 if any(pos in content.lower() for pos in ['positive', '1', 'pos']) else 0
            result = {
                'sentiment': sentiment,
                'confidence': 0.8,
                'reasoning': content
            }
            
            self._update_usage_stats(response)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        return {
            **self.usage_stats,
            'timestamp': datetime.now().isoformat()
        }    
    def predict_batch(
        self,
        reviews: List[str],
        template_name: str = 'batch_basic'
    ) -> List[Dict]:
        batch_input = "\n\n".join(
            f"Review {i}:\n{review}" 
            for i, review in enumerate(reviews)
        )
        
        if self.cache_responses:
            cache_key = self._compute_cache_key(template_name, [], batch_input)
            if cache_key in self.cache:
                return self._parse_batch_response(self.cache[cache_key])
        
        response = self._make_request(batch_input)
        
        if self.cache_responses:
            self.cache[cache_key] = response
            self._save_cache()
        
        return self._parse_batch_response(response)
    
    def _parse_batch_response(self, response: Any) -> List[Dict]:
        try:
            content = response.choices[0].message.content.strip()
            predictions = json.loads(content)
            
            if not isinstance(predictions, list):
                raise ValueError("Response is not a list")
                
            for pred in predictions:
                required_keys = {'id', 'sentiment', 'confidence'}
                if not all(key in pred for key in required_keys):
                    raise ValueError(f"Missing required keys in prediction: {pred}")
                if pred['sentiment'] not in [0, 1]:
                    raise ValueError(f"Invalid sentiment value: {pred['sentiment']}")
                
            return predictions
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {content}")
            raise
        except Exception as e:
            self.logger.error(f"Error parsing batch response: {str(e)}")
            raise
