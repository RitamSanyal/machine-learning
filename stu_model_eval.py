from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the pre-trained model (assuming it's saved as 'student_grade_model.h5')
model = load_model('stu_model.keras')


# Function to get user input and scale features
def get_user_input():
    study_time = float(input("Enter study time: "))
    failures = int(input("Enter number of failures: "))
    absences = int(input("Enter number of absences: "))
    G1 = float(input("Enter previous grade 1 (G1): "))
    G2 = float(input("Enter previous grade 2 (G2): "))

    # Preprocess the user input (similar to feature scaling done earlier)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform([[study_time, failures, absences, G1, G2]])
    return scaled_features


# Get user input
user_features = get_user_input()

# Predict the grade
predicted_grade = model.predict(user_features)[0][0]

# Print the predicted grade
print(f"Predicted final grade: {predicted_grade:.2f}")
