"""
Created on 2022-11-17 22:43:33
@author: Mitch Maletic

"""

import pickle
from fastapi import FastAPI

try:
    from schemas                            import (
                                                    FileStorage,
                                                    ModelResult,
                                                    MakePrediction
                                                    )
except ModuleNotFoundError:
    from scalable_pipeline.schemas          import (
                                                    FileStorage,
                                                    ModelResult,
                                                    MakePrediction
                                                    )
try:
    from src.train_model        import TrainModel
    from src.model_inference    import model_inference
except ModuleNotFoundError:
    from scalable_pipeline.src.train_model      import TrainModel
    from scalable_pipeline.src.model_inference  import model_inference

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global model, encoder, binarizer
    model       = pickle.load(open("./model/model.pkl", "rb"))
    encoder     = pickle.load(open("./model/encoder.pkl", "rb"))
    binarizer   = pickle.load(open("./model/binarizer.pkl", "rb"))
# end def

@app.get("/")
async def root():
    return {"Hello: Reviewer! I hope you like my code!"}
# end def

@app.post("/make_model/", response_model=ModelResult)
async def make_model(FileStorage: FileStorage):
    tm = TrainModel(FileStorage.file)

    return {"model": str(tm.model), "precision": tm.precision,
            "recall": tm.recall, "fbeta": tm.fbeta
            }
# end def

@app.post("/model_inference/")
async def make_prediction(MakePrediction: MakePrediction):
    preds = model_inference(MakePrediction)
    prediction = '>50k' if str(preds)=='1' else '<=50k'
    return {"model_inference": prediction}
# end def