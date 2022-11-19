"""
Created on 2022-11-17 22:43:33
@author: Mitch Maletic

"""

from fastapi import FastAPI

try:
    from schemas                            import (
                                                    FileStorage,
                                                    ModelInference
                                                    )
except ModuleNotFoundError:
    from scalable_pipeline.schemas          import (
                                                    FileStorage,
                                                    ModelInference
                                                    )
try:
    from src.train_model import TrainModel
except ModuleNotFoundError:
    from scalable_pipeline.src.train_model  import TrainModel

app = FastAPI()

@app.get("/")
async def root():
    return {"Hello: Reviewer! Hope you like my code!"}
# end def

@app.post("/model_inference/", response_model=ModelInference)
async def model_inference(FileStorage: FileStorage):
    tm = TrainModel(FileStorage.file)

    return {"model": str(tm.model), "precision": tm.precision,
            "recall": tm.recall, "fbeta": tm.fbeta
            }
# end def
