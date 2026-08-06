import pandas as pd
import joblib
df=pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
df.dropna(inplace=True)
df["Churn"]=df["Churn"].map({"Yes":1,"No":0})
df.drop("customerID",axis=1,inplace=True)
df=pd.get_dummies(df,drop_first=True)

x=df.drop("Churn",axis=1)
y=df["Churn"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

feature_names=x.columns.tolist()
joblib.dump(feature_names,"feature_names.pkl")
joblib.dump(model,"customer_Churn_model.pkl")
print("model saved successfully!")

from sklearn.tree import DecisionTreeClassifier

dt_model=DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train,y_train)
dt_pred=dt_model.predict(x_test)
dt_accuracy=accuracy_score(y_test,dt_pred)


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(x_train, y_train)

rf_pred = rf_model.predict(x_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nModel Comparison")
print("-----------------------")
print("Logistic Regression :", accuracy)
print("Decision Tree       :", dt_accuracy)
print("Random Forest       :", rf_accuracy)

