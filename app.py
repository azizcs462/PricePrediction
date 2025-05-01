import pickle
from flask import Flask,request,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
#Load the model
model = pickle.load(open('pricePredModel.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(np.array(list(data.values())).reshape(1,-1))

    df = pd.DataFrame([data])
    print(df)
    new_data = preprocessor.transform(df)
    print(new_data)


    # Transform input


    # # Predict (log price)
    log_price_prediction = model.predict(new_data)

    # # Inverse transform to get real price
    real_price_prediction = np.expm1(log_price_prediction)

    return jsonify({'prediction': float(real_price_prediction[0])})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data as a dictionary
        data = request.form.to_dict()
        print("Received data:", data)

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Ensure correct data types
        df = df.astype({
            "bathrooms": int,
            "bedrooms": int,
            "size": float
        })

        # Transform with preprocessor
        new_data = preprocessor.transform(df)

        # Predict and reverse log
        result = model.predict(new_data)
        real_price = np.expm1(result)

        print("Predicted price:", real_price[0])

        return render_template(
            "home.html",
            prediction_text=f"The prediction value is {real_price[0]:,.2f}"
        )

    except Exception as e:
        print("Error:", str(e))
        return render_template(
            "home.html",
            prediction_text=f"An error occurred: {str(e)}"
        )




    print(data)
    df = pd.DataFrame([data])
    new_data = preprocessor.transform(df)  # no fit, only transform!
    print("Transformed input:", new_data)

    # Predict using trained model
    result = model.predict(new_data)
    real_price = np.expm1(result)


if __name__ == "__main__":
    app.run(debug=True)






