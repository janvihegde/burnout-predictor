import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("Loading data...")
df = pd.read_csv("engineering_students_burnout_dataset.csv")

condition_stress = df['Stress_Level_1_10'] >= 6
condition_sleep = df['Sleep_Hours'] <=6
condition_study = df['Daily_Study_Hours'] >= 6

# Create the target column: 1 for Burnout Risk, 0 for No Risk
df['Burnout_Risk'] = ((condition_stress) & (condition_sleep | condition_study)).astype(int)

print(f"Dataset labelled. Risk distribution:\n{df['Burnout_Risk'].value_counts()}")

X = df.drop(['Student_ID', 'Burnout_Level_1_10', 'Burnout_Risk'], axis=1)
y = df['Burnout_Risk']

# 4. Preprocessing Pipeline
categorical_cols = ['Branch', 'Year', 'Interested_in_Core_Job']
numerical_cols = ['Daily_Study_Hours', 'Sleep_Hours', 'Weekly_Assignments', 
                  'Backlogs', 'Screen_Time_Hours', 'Stress_Level_1_10', 'Career_Clarity_1_10']

# Create transformers for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# 5. Initialize the Model (Random Forest)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 6. Train the model
print("Training the model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# 7. Evaluate
accuracy = model_pipeline.score(X_test, y_test)
print(f"Model trained successfully! Accuracy: {accuracy:.2f}")

# 8. Save the 'Brain' to a file
with open("burnout_model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

print("Model saved as 'burnout_model.pkl'. Ready for the API!")