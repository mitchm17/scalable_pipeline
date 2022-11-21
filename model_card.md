# Model Card

## Model Details
- Developed by Mitch Maletic
- Model created 17 November 2022
- Model version 1.0.0
- The model is a RandomForestClassifier

## Intended Use
- Intended to be used to get an accurate salary prediction based on Census data provided.
## Training Data
- The model was trained on publicly available Census Bureau data.
- The data contained nine categorical features and six numerical features.
- The data had to be pre-processed to remove the extra whitespaces found in the csv file. I also replaced any '?' with 'Unknown' values instead.
## Evaluation Data
- The model was evaluated using a split of the training dataset.
## Metrics
_Please include the metrics used and your model's performance on those metrics._
- The metrics used to analyze the RandomForestClassification model were
  - precision (0.736)
  - recall (0.632)
  - fbeta score, where beta=1 (0.680)
- It's clear from the metrics used that this model performs well at accurately predicting salaries.

## Ethical Considerations
- Risk Mitigations: there are no names or person's addresses located in the data, thus, none of the data can be tied to a specific person.
- Data: the dataset could be used to predict a person's salary based on their race or gender which could lead to possible nefarious misinformation uses.

## Caveats and Recommendations
- The dataset goes back many years and with the recent drastic changes of salary from 2019 to today, this data could be making predictions based on old data.
- The model can only make a prediction that, given a set of parameters, a salary will be 50k or less, or more than 50k. A different dataset may be able to provide a more detailed prediction if there are more salary options.