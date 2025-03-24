import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

file_path = r"C:\Users\Divya\OneDrive\Desktop\loan_approval_dataset.csv"    
df = pd.read_csv(file_path)
df = df.drop(columns=["loan_id"])
df.columns = df.columns.str.strip()
df = pd.get_dummies(df, columns=["education", "self_employed"], drop_first=True)

X = df.drop(columns=["loan_status"])
y = df["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=1)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

def predict_loan_approval(applicant_data):
    new_applicant = pd.DataFrame([applicant_data])
    new_applicant = pd.get_dummies(new_applicant, columns=["education", "self_employed"], drop_first=True)
    for col in X.columns:
        if col not in new_applicant:
            new_applicant[col] = 0  
    new_applicant = new_applicant[X.columns]
    loan_decision = rf_model.predict(new_applicant)[0]
    return loan_decision

applicant = {
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 9600000,
    "loan_amount": 29900000,
    "loan_term": 12,
    "cibil_score": 778,
    "residential_assets_value": 2400000,
    "commercial_assets_value": 17600000,
    "luxury_assets_value": 22700000,
    "bank_asset_value": 8000000
}

print(f"Loan Decision: {predict_loan_approval(applicant)}")