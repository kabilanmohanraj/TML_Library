# scripts/train.py
from tml_library import Trainer, initialize_train_pipeline, Evaluator

# Initialize components
X_train, X_test, y_train, y_test, model, param_grid = initialize_train_pipeline("./configs/config.json")
print("Model training...")

# Train and save the model
trainer = Trainer(model, param_grid)
best_model = trainer.train(X_train, y_train)
trainer.save_model("./best_model.joblib")
print("Model training complete and saved.")

evaluator = Evaluator(best_model)
metrics = evaluator.evaluate(X_test, y_test)
print(f"Evaluation Metrics:\n{metrics}")