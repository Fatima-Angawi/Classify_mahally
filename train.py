import numpy as np
from app.data.loader import load_dataset
from app.embeddings.embedder import Embedder
from app.models.trainer import Trainer
from app.models.evaluator import evaluate

df = load_dataset("data/combined_data.csv")

embedder = Embedder()
embeddings = embedder.encode(df["text"].astype(str).tolist())

X = np.array(embeddings)
y = df["label"].values

trainer = Trainer()
X_test, y_test = trainer.train(X, y)

evaluate(trainer.model, X_test, y_test)

trainer.save()
