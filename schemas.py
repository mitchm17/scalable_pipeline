"""
Created on 2022-11-17 22:40:25
@author: Mitch Maletic

"""

from pydantic import (
    BaseModel,
    FilePath
)

class FileStorage(BaseModel):
    file: FilePath

    class Config:
        schema_extra = {
            "example": {
                "file": "/path/to/data.csv"
            }
        }
# end class

class ModelInference(BaseModel):
    model: str
    precision: float
    recall: float = "n/a"
    fbeta: float = "n/a"

    class Config:
        schema_extra = {
            "example": {
                "model": "RandomForestClassifier(random_state=42)",
                "precision": 0.999999999999999,
                "recall": 0.999999999999999,
                "fbeta": 0.999999999999999,
            }
        }
# end class