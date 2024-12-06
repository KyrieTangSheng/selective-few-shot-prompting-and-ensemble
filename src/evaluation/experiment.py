from dataclasses import dataclass
from typing import Dict, List
import json
from pathlib import Path
import logging
from .evaluator import SystemEvaluator
from utils.data_loader import IMDBDataLoader
from embedding.encoder import TextEncoder
from database.vector_store import VectorStore
from utils.device_utils import get_device
from ensemble.config import EnsembleConfig, EnsembleMemberConfig
from ensemble.member import EnsembleMember 
from ensemble.ensemble import EnsemblePredictor
from selection.selector import ExampleSelector, BaselineSelector
from prompting.template_manager import TemplateManager
from prompting.prompt_builder import PromptBuilder
from llm.interface import LLMInterface
from llm.config import LLMConfig


@dataclass
class ExperimentConfig:
    """Basic configuration for single strategy experiment"""
    name: str
    selector_strategy: str
    selector_params: Dict
    template_name: str
    llm_temperature: float
    test_size: int = -1  # -1 means full test set
    
class ExperimentRunner:
    def __init__(
        self,
        api_key: str,
        base_path: str = "./experiments"
    ):
        self.api_key = api_key
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def run_single_strategy(self, config: ExperimentConfig) -> Dict:
        """Run experiment with single strategy configuration"""
        self.logger.info(f"Starting experiment: {config.name}")
        
        encoder, vector_store, train_df, test_df = self._setup_base_components()
        
        member = self._create_ensemble_member(
            name=f"{config.selector_strategy}_{config.template_name}",
            selector_strategy=config.selector_strategy,
            selector_params=config.selector_params,
            template_name=config.template_name,
            llm_temperature=config.llm_temperature
        )
        
        ensemble = self._create_single_member_ensemble(member)
        
        evaluator = SystemEvaluator(
            encoder=encoder,
            vector_store=vector_store,
            ensemble_predictor=ensemble
        )
        
        if config.test_size > 0:
            test_subset = test_df.head(config.test_size)
        else:
            test_subset = test_df
            
        self.logger.info(f"Evaluating on {len(test_subset)} examples...")
        results = evaluator.evaluate(test_subset, progress_bar=True)
        
        self._save_results(config.name, results)
        
        return results
    
    def _setup_base_components(self):
        """Initialize base components needed for experiments."""
        device = get_device(force_cpu=False)
        
        loader = IMDBDataLoader("./dataset/IMDB_Dataset.csv")
        train_df, test_df = loader.load_data()
        
        encoder = TextEncoder(device=str(device))
        
        train_subset = train_df
        embeddings = encoder.encode(train_subset['review'].tolist())
        
        vector_store = VectorStore(dimension=encoder.embedding_dim)
        original_data = [
            {'text': row['review'], 'sentiment': row['sentiment']} 
            for _, row in train_subset.iterrows()
        ]
        vector_store.add_vectors(embeddings, original_data)
        
        return encoder, vector_store, train_df, test_df
    
    def _create_ensemble_member(self, name: str, selector_strategy: str, selector_params: Dict, template_name: str, llm_temperature: float):
        template_manager = TemplateManager()
        
        if selector_strategy in {'zero_shot', 'few_shot'}:
            selector = BaselineSelector(selector_strategy, **selector_params)
        else:
            selector = ExampleSelector(selector_strategy, **selector_params)
        
        prompt_builder = PromptBuilder(template_manager)
        llm = LLMInterface(
            self.api_key,
            config=LLMConfig(temperature=llm_temperature)
        )
        
        return EnsembleMember(name, selector, prompt_builder, llm)
    
    def _create_single_member_ensemble(self, member: EnsembleMember):
        ensemble_config = EnsembleConfig(
            members=[member.name],
            voting_method='majority',
            min_confidence=0.5,
            cache_predictions=True
        )
        
        return EnsemblePredictor([member], ensemble_config)
    
    def _save_results(self, name: str, results: Dict):
        exp_dir = self.base_path / name
        exp_dir.mkdir(exist_ok=True)
        
        with open(exp_dir / "results.json", "w") as f:
            json.dump(results, f)
    
    def run_ensemble_strategy(self, configs: List[ExperimentConfig], test_size: int = -1):
        logging.info(f"Starting ensemble experiment with {len(configs)} members")
        encoder, vector_store, train_df, test_df = self._setup_base_components()

        members = []
        for config in configs:
            member = self._create_ensemble_member(
                name=config.name,
                selector_strategy=config.selector_strategy,
                selector_params=config.selector_params,
                template_name=config.template_name,
                llm_temperature=config.llm_temperature
            )
            members.append(member)
        
        member_configs = []
        for config in configs:
            config_dict = vars(config)
            config_dict.pop('test_size')  
            member_configs.append(EnsembleMemberConfig(**config_dict))
        
        ensemble_config = EnsembleConfig(
            members=member_configs,
            voting_method='weighted',
            min_confidence=0.6
        )
        
        ensemble = EnsemblePredictor(members, ensemble_config)
        
        test_subset = test_df.head(test_size)
        
        evaluator = SystemEvaluator(
            encoder=encoder,
            vector_store=vector_store,
            ensemble_predictor=ensemble
        )
        
        results = evaluator.evaluate(test_subset, progress_bar=True)
        self._save_results("ensemble",results)
        return results
    