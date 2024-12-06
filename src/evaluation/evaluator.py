from typing import List, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import json
from datetime import datetime
import os

class SystemEvaluator:
    def __init__(
        self,
        encoder,
        vector_store,
        ensemble_predictor,
        output_dir: str = "./results"
    ):
        self.encoder = encoder
        self.vector_store = vector_store
        self.ensemble = ensemble_predictor
        self.output_dir = output_dir
        self.results = {}
    
    def evaluate(
        self,
        test_df: pd.DataFrame,
        progress_bar: bool = True
    ) -> Dict:
        """Evaluate system on test data."""
        predictions = []
        confidences = []
        processing_times = []
        total_cost = 0
        
        iterator = tqdm(
            test_df.iterrows(),
            total=len(test_df),
            desc="Evaluating",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        ) if progress_bar else test_df.iterrows()
        
        for _, row in iterator:
            result = self._process_single(row)
            predictions.append(result['prediction'])
            confidences.append(result['confidence'])
            processing_times.append(result['processing_time'])
            total_cost += result.get('cost', 0)
        
        self._calculate_metrics(predictions, test_df['sentiment'].tolist(), confidences)
        
        self.results['timing_stats'] = {
            'total_time': sum(processing_times),
            'average_time_per_example': np.mean(processing_times)
        }
        self.results['cost_analysis'] = {
            'total_cost': total_cost,
            'average_cost_per_example': total_cost / len(test_df)
        }
        
        return self.results
    
    def _process_single(self, row) -> Dict:
        """Process a single test example."""
        review = row['review']
        true_label = row['sentiment']
        
        start_time = time.time()
        review_embedding = self.encoder.encode(review)
        search_results = self.vector_store.search(review_embedding, k=20)
        
        member_prediction = self.ensemble.predict(
            review,
            search_results,
            true_label
        )
        
        processing_time = time.time() - start_time
        
        first_member = self.ensemble.members[0]
        selection_results = first_member.selector.select(search_results, k=3)
        prompt_result = first_member.prompt_builder.build_prompt(
            selection_results.examples,
            review,
            template_name=first_member.name.split('_')[1]
        )
        # print("\nPrompt used:")
        # print("-"*80)
        # print(prompt_result.prompt)
        # print("-"*80)
        
        return {
            'true_label': true_label,
            'prediction': member_prediction.prediction, 
            'confidence': member_prediction.confidence,  
            'processing_time': processing_time,
            'cost': 0.0,  
            'metadata': {
                'reasoning': member_prediction.metadata.get('reasoning'),
                'selection_scores': member_prediction.metadata.get('selection_scores')
            }
        }
    
    def _calculate_metrics(
        self,
        predictions: List[int],
        true_labels: List[int],
        confidences: List[float]
    ):
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            predictions,
            average='binary'
        )
        
        confidence_bins = np.linspace(0, 1, 11)
        conf_accuracy = []
        for i in range(len(confidence_bins)-1):
            mask = (np.array(confidences) >= confidence_bins[i]) & \
                   (np.array(confidences) < confidence_bins[i+1])
            if np.any(mask):
                bin_acc = accuracy_score(
                    np.array(true_labels)[mask],
                    np.array(predictions)[mask]
                )
                conf_accuracy.append((confidence_bins[i], bin_acc))
        
        self.results['metrics'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confidence_calibration': conf_accuracy
        }
    
    def _analyze_components(self):
        member_stats = self.ensemble.get_performance_stats()['member_stats']
        
        strategy_performance = {}
        for member_name, stats in member_stats.items():
            strategy = member_name.split('_')[0] 
            if strategy not in strategy_performance:
                strategy_performance[strategy] = []
            strategy_performance[strategy].append(stats['accuracy'])
        
        self.results['component_analysis'] = {
            'member_performance': member_stats,
            'strategy_performance': {
                strategy: np.mean(accuracies)
                for strategy, accuracies in strategy_performance.items()
            }
        }
    
    def _save_results(self):
        """Save evaluation results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"eval_results_{timestamp}.json")
        
        results_copy = json.loads(json.dumps(self.results, default=str))
        
        with open(output_file, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"Results saved to {output_file}")