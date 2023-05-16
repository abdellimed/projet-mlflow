import numpy as np
import bentoml
import typing

import numpy as np
import pandas as pd
from pydantic import BaseModel

import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray

runner = bentoml.mlflow.get("comment-classifier:latest").to_runner()
svc = bentoml.Service('comment-classifier', runners=[runner])

class Features(BaseModel):
    comment: str
    # Optional field
   # request_id: typing.Optional[int]
    # Use custom Pydantic config for additional validation options
   # class Config:
      #  extra = "forbid"
input_spec = JSON(pydantic_model=Features)
@svc.api(input=input_spec, output=JSON())
async def predict(input_text: Features):
    input_df = pd.DataFrame([input_text.dict()])
    return {'res':runner.predict.async_run(input_df)[0]}
