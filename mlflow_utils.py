import mlflow

def init_mlflow_experiment(experiment_name, run_name):
    """
    1. Sets the experiment (group of runs).
    2. Enables autologging (automatically tracks params, metrics, models).
    3. Starts the run with a CUSTOM name (no dummy names!).
    """
    # Set the experiment. If it doesn't exist, MLflow creates it.
    mlflow.set_experiment(experiment_name)
    
    # Enable autologging. This tells MLflow: "Watch my sklearn models and log everything."
    mlflow.sklearn.autolog()
    
    # Start the run manually so we can give it a specific name.
    # We return the 'run' object so we can use it in a 'with' block.
    return mlflow.start_run(run_name=run_name)