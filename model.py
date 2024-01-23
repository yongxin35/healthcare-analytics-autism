import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle


# Create the best model
def create_best_model():
    df = pd.read_csv("Autism-Child-Data-Cleaned_.csv")

    X = df.drop('Class/ASD', axis=1)
    y = df['Class/ASD']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    classificationReport = classification_report(y_test, y_pred)

    print(f"Accuracy of Logistic Regression Model: {accuracy}")
    print(f"Classification Report of Logistic Regression Model: \n{classificationReport}")

    # Testing
    print(model.predict_proba(X_test[0:1]))
    print(X_test[0:1])

    print(y_test[0:1])

    return model


# Save the model as a pickle (binary) file, so it will be at the backend of our app.
# This will avoid our app from train, test, split the model, everytime the user enters a new input. Which will be very time and power consuming.
def main():
    model = create_best_model()

    # Save and export the model into a pickle file
    pickle.dump(model, open('model.pkl', 'wb'))


if __name__ == '__main__':
    main()
