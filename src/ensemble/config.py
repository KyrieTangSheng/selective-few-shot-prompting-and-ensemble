from dataclasses import dataclass
from typing import List, Dict, Optional
from selection.selector import ExampleSelector
from prompting.template_manager import PromptTemplate

@dataclass
class EnsembleMemberConfig:
    name: str
    selector_strategy: str
    selector_params: Dict
    template_name: str
    llm_temperature: float
    weight: float = 1.0

@dataclass
class EnsembleConfig:
    members: List[EnsembleMemberConfig]
    voting_method: str = 'weighted' 
    min_confidence: float = 0.6
    cache_predictions: bool = True