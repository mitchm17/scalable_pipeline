"""
Created on 2022-11-17 22:40:25
@author: Mitch Maletic

"""
from typing     import Optional, Union
from pydantic   import (
    BaseModel,
    Field,
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

class ModelResult(BaseModel):
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

class MakePrediction(BaseModel):
    age:            Union[int, list]
    workclass:      Union[str, list]
    fnlgt:          Union[int, list]
    education:      Union[str, list]
    education_num:  Union[int, list] = Field(alias="education-num")
    marital_status: Union[str, list] = Field(alias="marital-status")
    occupation:     Union[str, list]
    relationship:   Union[str, list]
    race:           Union[str, list]
    sex:            Union[str, list]
    capital_gain:   Union[int, list] = Field(alias="capital-gain")
    capital_loss:   Union[int, list] = Field(alias="capital-loss")
    hours_per_week: Union[int, list] = Field(alias="hours-per-week")
    native_country: Union[str, list] = Field(alias="native-country")

    class Config:
        schema_extra = {
            "example": {
                "age": [39],
                "workclass": ["State-gov"],
                "fnlgt": [77516],
                "education": ["Bachelors"],
                "education-num": [13],
                "marital-status": ["Never-married"],
                "occupation": ["Adm-clerical"],
                "relationship": ["Not-in-family"],
                "race": ["White"],
                "sex": ["Male"],
                "capital-gain": [2174],
                "capital-loss": [0],
                "hours-per-week": [40],
                "native-country": ["United-States"]
            }
        }
    # end Config class
# end class