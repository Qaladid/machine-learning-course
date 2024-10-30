import pickle
from flask import Flask, request, jsonify

# Load the DictVectorizer and model separately
dv_file = 'dv.bin'  # Path to the DictVectorizer file
model_file = 'model1.bin'  # Path to the model file

# Load the DictVectorizer
with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

# Load the model
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('churn')

@app.route('/app', methods=['POST'])
def app():  
    client = request.get_json()
    print("Received client data:", client)  # Debugging line

    if not client:
        return jsonify({"error": "No input data provided"}), 400

    try:
        # Transform the client data into the correct format
        X = dv.transform([client])
        y_pred = model.predict_proba(X)[0, 1]
        churn = y_pred >= 0.5

        result = {
            'churn_probability': float(y_pred),
            'churn': bool(churn)    
        }

        return jsonify(result)
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696, debug=True)
