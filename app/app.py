from flask import Flask, render_template, request
from recommender import generate_recommendations
from model_loader import movie_titles
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    error_message = None

    if request.method == 'POST':
        user_id = request.form.get('user_id')

        if user_id:
            try:
                recommendations = generate_recommendations(user_id, n=10)
                if not recommendations:
                    error_message = "No recommendations found for this user."
            except Exception as e:
                error_message = f"Error: {str(e)}"
        else:
            error_message = "Please enter a User ID."

    return render_template('index.html', recommendations=recommendations, movie_titles=movie_titles, error=error_message)

@app.route('/metrics')
def metrics():
    from surprise import accuracy
    from model_loader import model, data

    # Rebuild train/test to evaluate again (optional for display)
    trainset = data.build_full_trainset()
    testset = trainset.build_testset()
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)

    return render_template('metrics.html', rmse=rmse)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)
