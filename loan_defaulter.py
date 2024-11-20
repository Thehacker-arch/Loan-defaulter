import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


try:
    loan_data = pd.read_csv('loan.csv')
except FileNotFoundError:
    print("Error: 'loan.csv' not found. Please upload the file.")
    exit()


X = loan_data.drop('default', axis=1)
y = loan_data['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Model Accuracy: {accuracy * 100:.2f}%")


def predict_pd(loan_features):
    loan_df = pd.DataFrame([loan_features], columns=X_train.columns) 
    loan_df = loan_df.fillna(0)
    pd_probability = model.predict_proba(loan_df)[:,1][0]
    return pd_probability

def expected_loss(loan_features, loan_amount, recovery_rate=0.10):
    pd_probability = predict_pd(loan_features)
    expected_loss_value = pd_probability * loan_amount * (1 - recovery_rate)
    return expected_loss_value


loan_features_example = {
    'customer_id': 123,
    'credit_lines_outstanding': 3,
    'loan_amt_outstanding': 50000,
    'total_debt_outstanding': 100000,
    'income': 400000,
    'years_employed': 3,
    'fico_score': 700
    }

loan_amount = 305000
el = expected_loss(loan_features_example, loan_amount)
print(f"Expected Loss: ${el}")