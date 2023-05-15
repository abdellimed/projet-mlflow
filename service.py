import numpy as np
import bentoml

runner = bentoml.mlflow.get("comment-classifier:latest").to_runner()

svc = bentoml.Service('comment-classifier', runners=[runner])


@svc.api(input=bentoml.io.Text(), output=bentoml.io.NumpyNdarray())
def predict(input_text: str):
    return runner.predict.run([input_text])[0]
