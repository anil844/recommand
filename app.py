from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the model and vectorizer from the pickle file
with open('medicine_recommendation_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

vectorizer = model_data['vectorizer']
tfidf_matrix = model_data['tfidf_matrix']
data = model_data['data']

# Function to get medicine recommendations based on symptoms
def recommend_medicine(symptoms):
    symptoms_vector = vectorizer.transform([symptoms])
    cosine_sim = cosine_similarity(symptoms_vector, tfidf_matrix)
    similar_medicines = np.argsort(-cosine_sim[0])
    # Convert the recommendations to a list
    return data['Medicine'].iloc[similar_medicines].head(5).tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    symptoms = request.form['symptoms']
    recommendations = recommend_medicine(symptoms)
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
