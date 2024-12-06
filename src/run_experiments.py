from evaluation.experiment import ExperimentRunner, ExperimentConfig

def main():
    # API key
    api_key = "Your API key here"
    
    # Initialize runner
    runner = ExperimentRunner(api_key)
    
    # Add baseline configurations
    baseline_configs = [
        ExperimentConfig(
            name="zeroshot_zeroshot",
            selector_strategy="zeroshot",
            selector_params={},
            template_name="zeroshot",
            llm_temperature=0.0,
            test_size=-1
        ),
        ExperimentConfig(
            name="fewshot_fewshot",
            selector_strategy="fewshot",
            selector_params={},  
            template_name="fewshot",
            llm_temperature=0.0,
            test_size=-1
        )
    ]
    # Basic strategy configurations
    configs = [
        ExperimentConfig(
            name="diversity_basic",
            selector_strategy="similarity",
            selector_params={},
            template_name="basic",
            llm_temperature=0.0,
            test_size=-1
        ),
        ExperimentConfig(
            name="similarity_basic",
            selector_strategy="similarity",
            selector_params={},
            template_name="basic",
            llm_temperature=0.0,
            test_size=-1
        ),
        ExperimentConfig(
            name="hybrid_basic",
            selector_strategy="hybrid", 
            selector_params={'diversity_weight': 0.3},
            template_name="basic",
            llm_temperature=0.1,
            test_size=-1
        ),
    ]
    
    # Run baseline experiments
    print("\n=== Running Baseline Experiments ===")
    for config in baseline_configs:
        print(f"\nRunning {config.name}...")
        results = runner.run_single_strategy(config)
        print(f"Accuracy: {results['metrics']['accuracy']:.3f}")
        print(f"F1 Score: {results['metrics']['f1_score']:.3f}")
    
    # Run individual experiments
    print("\n=== Running Individual Experiments ===")
    for config in configs:
        results = runner.run_single_strategy(config)
        print(f"\nResults for {config.name}:")
        print(f"Accuracy: {results['metrics']['accuracy']:.3f}")
        print(f"F1 Score: {results['metrics']['f1_score']:.3f}")
    
    # Run ensemble experiment
    print("\n=== Running Ensemble Experiment ===")
    ensemble_results = runner.run_ensemble_strategy(
        configs=configs,
        test_size=100
    )
    print("\nEnsemble Results:")
    print(f"Accuracy: {ensemble_results['metrics']['accuracy']:.3f}")
    print(f"F1 Score: {ensemble_results['metrics']['f1_score']:.3f}")


if __name__ == "__main__":
    main() 