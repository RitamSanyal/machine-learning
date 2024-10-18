from sklearn.preprocessing import StandardScaler
from keras._tf_keras.keras.models import load_model
import pandas as pd

# Load the pre-trained model
model = load_model('stu_model.keras')

# Load the training data to get the scaler
data = pd.read_csv('StudentPerformanceFactors.csv')
features = ['Hours_Studied', 'Sleep_Hours', 'Attendance', 'Previous_Scores']
X_train = data[features]

# Fit the scaler on the training data
scaler = StandardScaler()
scaler.fit(X_train)


# Function to scale user input features
def scale_user_input(user_input, scaler):
    scaled_features = scaler.transform([user_input])
    return scaled_features


def get_user_input():
    Hours_Studied = float(input("Enter study time: "))
    Sleep_Hours = int(input("Enter number of Sleep_Hours: "))
    Attendance = int(input("Enter number of Attendance: "))
    Attendance = float(input("Enter previous grade 1 (G1): "))
    # Exam_Score = float(input("Enter previous grade 2 (G2): "))
    return [Hours_Studied, Sleep_Hours, Attendance, Attendance]


# Get user input
user_input = get_user_input()

# Scale user input features
scaled_input = scale_user_input(user_input, scaler)

# Predict the grade5
predicted_grade = model.predict(scaled_input)[0][0]

# Print the predicted grade
print(f"Predicted final grade: {predicted_grade:.2f}")
