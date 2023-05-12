import numpy as np

import bentoml
from bentoml.io import NumpyNdarray

model_runner = bentoml.mlflow.get("projet_test:latest").to_runner()

svc = bentoml.Service("projet_test_service", runners=[model_runner])

input_spec = NumpyNdarray(
    dtype="int",
    enforce_dtype=True,
    shape=(-1, 6),
    enforce_shape=True,
)


@svc.api(input=input_spec, output=NumpyNdarray())
async def classify(input_series: np.ndarray) -> np.ndarray:
    return await lgb_iris_runner.predict.async_run(input_series)
