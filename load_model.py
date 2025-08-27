import joblib
import pandas as pd

# Load saved objects
model = joblib.load("life_satisfaction_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")
X_columns = joblib.load("X_columns.pkl")

def predict_life_satisfaction(user_input: dict):
    input_df = pd.DataFrame([user_input])
    input_encoded = pd.get_dummies(input_df).reindex(columns=X_columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    pred = model.predict(input_scaled)
    return le.inverse_transform(pred)[0]

if __name__ == "__main__":
    # Ask all 19 questions
    user_input = {}
    user_input["age"] = int(input("1. What is your age? "))
    user_input["A2"] = input("2. How would you rate your health generally? ")
    user_input["C1"] = input("3. Do you suffer from a long-term physical health problem or disability? ")
    user_input["D2"] = input("4. Are you depressed? ")
    user_input["D8"] = input("5. Do you see yourself as a person who worries a lot? ")
    user_input["D10"] = input("6. Do you see yourself as a person who is emotionally stable, and not easily excited? ")
    user_input["D15"] = input("7. Do you see yourself as a person who prepares plans and implements them? ")
    user_input["job"] = input("8. Do you hold a job? ")
    user_input["F15"] = int(input("9. On a scale from 0-10, how content are you with your job? "))
    user_input["M8"] = input("10. How would you rate your current finances? ")
    user_input["E17"] = input("11. Who do you primarily talk to about personal and serious problems? ")
    user_input["G1"] = input("12. Do you have a spouse/partner/boy/girlfriend? ")
    user_input["J2"] = input("13. How often have you spent time with other relatives in the past year? ")
    user_input["J4"] = input("14. In the past year, how often have you spent time with acquaintances? ")
    user_input["J9"] = input("15. In the past year, how often have you been to the cinema, a concert, or the theater? ")
    user_input["E1"] = float(input("16. How tall are you (cm)? "))
    user_input["E2"] = float(input("17. How much do you weigh (kg)? "))
    user_input["J17"] = input("18. How often have you been abroad on holiday/family visits in the past year? ")
    user_input["M6"] = float(input("19. In the past year, how much have you spent on medicine/supplements? "))

    # Prediction
    prediction = predict_life_satisfaction(user_input)
    print("\n✅ Predicted Life Satisfaction:", prediction)
