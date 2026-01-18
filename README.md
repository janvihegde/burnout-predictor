# Academic Burnout Prediction System

## Project Overview
The Academic Burnout Prediction System is a full-stack web application designed to assess student mental health risks. By analyzing daily behavioral data—such as study hours, sleep patterns, and stress levels—the system uses Machine Learning to predict the likelihood of academic burnout.

The project solves the problem of undetectable early-stage burnout by providing an accessible, automated screening tool that offers immediate feedback and recommendations.

## Features
- **Real-time Prediction:** Instant assessment of burnout risk using a trained Machine Learning model.
- **Behavioral Analysis:** Evaluates multiple factors including sleep, study intensity, screen time, and academic backlog.
- **Full-Stack Architecture:** Decoupled React frontend and Python FastAPI backend.
- **Data Logging:** Capable of storing prediction history in MongoDB for trend analysis.
- **Responsive UI:** Clean, modern interface built with Tailwind CSS.

## Technology Stack

### Backend
- **Python:** Core programming language.
- **FastAPI:** High-performance web framework for building APIs.
- **Scikit-Learn:** Machine learning library used for the Random Forest Classifier.
- **Pandas/NumPy:** Data manipulation and preprocessing.

### Frontend
- **React.js:** JavaScript library for building the user interface.
- **Tailwind CSS:** Utility-first CSS framework for styling.
- **Axios:** HTTP client for API communication.

## System Architecture
1. **User Input:** The user submits data via the React frontend form.
2. **API Request:** Axios sends a POST request to the FastAPI backend.
3. **ML Inference:** The backend pre-processes the data and feeds it into the `burnout_model.pkl` (Random Forest) model.
4. **Result Generation:** The model returns a binary classification (Risk/No Risk) and a probability score.
5. **Response:** The frontend displays the result with tailored recommendations.

## Machine Learning Approach
The model uses a **Weak Supervision** strategy. Since real-world "burnout" labels are subjective, the dataset was labeled using a rule-based logic derived from psychological indicators:
- **Condition:** High Stress AND (Low Sleep OR Excessive Study Hours).
- **Model:** Random Forest Classifier.
- **Accuracy:** The model achieves high accuracy in detecting these specific behavioral patterns.

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Node.js and npm
- Git

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/burnout-prediction-system.git](https://github.com/YOUR_USERNAME/burnout-prediction-system.git)
cd burnout-prediction-system
