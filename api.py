from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()


class university_info(BaseModel):
    no_of_student: int
    no_of_student_per_staff: float
    international_student: float
    teaching_score: float
    research_score: float
    citations_score: float
    industry_income_score: float
    international_outlook_score: float
    female_ratio: float
    male_ratio: float


ATTRIBUTE_TO_COLUMN_MAP = {
    "no_of_student": "No of student",
    "no_of_student_per_staff": "No of student per staff",
    "international_student": "International Student",
    "teaching_score": "Teaching Score",
    "research_score": "Research Score",
    "citations_score": "Citations Score",
    "industry_income_score": "Industry Income Score",
    "international_outlook_score": "International Outlook Score",
    "female_ratio": "Female Ratio",
    "male_ratio": "Male Ratio",
}
# Load the model
with open('university_ranking_model.pkl', 'rb') as f_model:
    model = pickle.load(f_model)

# Load the scaler
with open('university_ranking_scaler.pkl', 'rb') as f_scaler:
    scaler = pickle.load(f_scaler)


@app.post('/')
async def scoring_endpoint(item: university_info):
    # scoring code
    data_dict = {ATTRIBUTE_TO_COLUMN_MAP[k]: v for k, v in item.dict().items()}
    df = pd.DataFrame([data_dict])
    x = scaler.transform(df)
    yhat = model.predict(x)
    return {'predictions': yhat[0]}
