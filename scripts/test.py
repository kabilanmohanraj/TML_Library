from tml_library.evaluator import Evaluator
from tml_library.utils import initialize_test_pipeline

# Initialize components from the configuration file
X_test, y_test, model, logger = initialize_test_pipeline("./configs/config.json")

# Evaluate model
evaluator = Evaluator(model)
metrics = evaluator.evaluate(X_test, y_test)
evaluator.plot_roc_curve(X_test, y_test)

# Print metrics
print("Evaluation Metrics:", metrics)
logger.info(f"Evaluation Metrics: {metrics}")
