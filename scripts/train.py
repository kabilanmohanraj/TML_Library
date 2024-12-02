# scripts/train.py
from tml_library import Trainer, initialize_train_pipeline

# Initialize components
X, y, model, param_grid, logger = initialize_train_pipeline("./configs/config.json")

# Train and save the model
trainer = Trainer(model, param_grid)
best_model = trainer.train(X, y)
trainer.save_model("./best_model.joblib")
logger.info("Model training complete and saved.")
