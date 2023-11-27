import joblib

import engine



def use_model():
    model = joblib.load('municipal_crime_classifier.pkl')
    df, X, y = engine.read_data()

    names = [df['Settlement_Council'][i] for i in range(0, len(df), 21)]

    for i in range(0, len(X)):
        if int(model.predict([X[i]])[0]) != int(y[i]):
            print(f"{names[i]}: ", int(model.predict([X[i]])[0]), int(y[i]))

