# University Ranking Prediction

This project demonstrates the process of building a machine learning model to predict university rankings based on various features. Additionally, it provides a FastAPI-based API for making predictions with the trained model.


**API Endpoint:**`https://simple-ml-api-4b6b50e31b43.herokuapp.com/`

Send a POST request with the following JSON object to get predictions:

{
    "no_of_student":32000,
    "no_of_student_per_staff":23.5,
    "international_student":0.15,
    "teaching_score":97.4,
    "research_score":96.8,
    "citations_score":92.5,
    "industry_income_score":94.3,
    "international_outlook_score":97.1,
    "female_ratio":52.5,
    "male_ratio":47.5
}

## Table of Contents

* [Overview](https://chat.openai.com/c/06029009-b568-41e3-822a-96b77a306b9f#overview)
* [Getting Started](https://chat.openai.com/c/06029009-b568-41e3-822a-96b77a306b9f#getting-started)
  * [Prerequisites](https://chat.openai.com/c/06029009-b568-41e3-822a-96b77a306b9f#prerequisites)
  * [Installation](https://chat.openai.com/c/06029009-b568-41e3-822a-96b77a306b9f#installation)
* [Data Preprocessing and Model Training](https://chat.openai.com/c/06029009-b568-41e3-822a-96b77a306b9f#data-preprocessing-and-model-training)
* [API Usage](https://chat.openai.com/c/06029009-b568-41e3-822a-96b77a306b9f#api-usage)
* [Contributing](https://chat.openai.com/c/06029009-b568-41e3-822a-96b77a306b9f#contributing)
* [License](https://chat.openai.com/c/06029009-b568-41e3-822a-96b77a306b9f#license)

## Overview

The goal of this project is to predict university rankings using machine learning. The steps involved in this project include:

1. **Data Preprocessing:** The project starts with data preprocessing. The dataset, "World University Rankings 2023.csv," is loaded and cleaned to remove missing values and format columns correctly.
2. **Model Training:** A Linear Regression model is trained on the preprocessed dataset. The model is used to predict university rankings based on features like the number of students, teaching scores, research scores, etc.
3. **Scalability:** The StandardScaler is used to standardize the input features for making predictions.
4. **API Development:** The FastAPI framework is used to create a simple API for university ranking predictions. The API takes input in the form of a JSON object, processes it, and returns a prediction.

## Getting Started

### Prerequisites

Before running this project, ensure you have the following prerequisites:

* Python 3.x
* Jupyter Notebook (optional)
* Heroku CLI (optional for deployment)

### Installation

1. Clone the repository:
   <pre><div class="bg-black rounded-md mb-4"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>Copy code</button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-bash">git clone https://github.com/StevenHHB/simple_ml_api.git
   </code></div></div></pre>
2. Install the required Python packages:
   <pre><div class="bg-black rounded-md mb-4"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>Copy code</button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-bash">pip install -r requirements.txt
   </code></div></div></pre>

## Data Preprocessing and Model Training

1. **Data Preprocessing:** The data preprocessing step is essential to clean and format the dataset. It involves handling missing values, converting data types, and creating appropriate features. This is done in the `preprocess_data` function.
2. **Model Training:** The machine learning model, a Linear Regression model, is trained on the preprocessed dataset. This model is used to predict university rankings. You can find the code for training the model in a Jupyter Notebook or Python script.
3. **Standardization:** Input features are standardized using the `StandardScaler` to ensure consistency when making predictions.
4. **Model Persistence:** Both the trained model and the scaler are saved to pickle files (`university_ranking_model.pkl` and `university_ranking_scaler.pkl`, respectively) for later use.

## API Usage

Once the model and scaler are trained and saved, you can start the FastAPI-based API to make university ranking predictions.

1. Run the FastAPI application:
   <pre><div class="bg-black rounded-md mb-4"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>Copy code</button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-bash">uvicorn api:app --host 0.0.0.0 --port 8000 --reload
   </code></div></div></pre>
2. Access the API at `http://localhost:8000` using an API client or a tool like `curl`.
3. Send a POST request with the following JSON object to get predictions:
   <pre><div class="bg-black rounded-md mb-4"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>json</span><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>Copy code</button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-json">{
       "no_of_student": 32000,
       "no_of_student_per_staff": 23.5,
       "international_student": 0.15,
       "teaching_score": 97.4,
       "research_score": 96.8,
       "citations_score": 92.5,
       "industry_income_score": 94.3,
       "international_outlook_score": 97.1,
       "female_ratio": 52.5,
       "male_ratio": 47.5
   }
   </code></div></div></pre>
4. You will receive a JSON response with the prediction:
   <pre><div class="bg-black rounded-md mb-4"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>json</span><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>Copy code</button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-json">{
       "predictions": 15.23
   }
   </code></div></div></pre>

## Contributing

Contributions are welcome! If you'd like to improve this project or fix any issues, please submit a pull request. For major changes, please open an issue first to discuss the proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](https://chat.openai.com/c/LICENSE) file for details.
