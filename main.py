
from flask import Flask, render_template, request
import numpy as np
import pickle
import io
import base64
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load your model (replace with preferred one)
model = pickle.load(open('model_lr.pkl', 'rb'))

@app.route("/")
def about():
    return render_template('home.html')

@app.route("/about")
def home():
    return render_template('about.html')

@app.route("/predict")
def home1():
    return render_template('predict.html')

@app.route("/submit")
def home2():
    return render_template('submit.html')

@app.route("/pred", methods=['POST'])
def predict():
    # Gather form data
    quarter = request.form['quarter']
    department = request.form['department']
    day = request.form['day']
    team = request.form['team']
    targeted_productivity = request.form['targeted_productivity']
    smv = request.form['smv']
    over_time = request.form['over_time']
    incentive = request.form['incentive']
    idle_time = request.form['idle_time']
    idle_men = request.form['idle_men']
    no_of_style_change = request.form['no_of_style_change']
    no_of_workers = request.form['no_of_workers']
    month = request.form['month']

    total = [[int(quarter), int(department), int(day), int(team),
              float(targeted_productivity), float(smv), int(over_time), int(incentive),
              float(idle_time), int(idle_men), int(no_of_style_change), float(no_of_workers), int(month)]]

    # Prediction and message
    prediction = model.predict(total)[0]
    if prediction <= 0.3:
        category = 'The employee is Averagely Productive.'
    elif prediction <= 0.8:
        category = 'The employee is Medium Productive.'
    else:
        category = 'The employee is Highly Productive.'
    
    result_text = f"{category} (Prediction Score: {prediction * 100:.2f}%)"

    
    # Graph generation
    graphs = []
    def create_plot():
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        graphs.append(image_base64)
        plt.close()

    # Bar Chart
    plt.figure()
    categories = ['Targeted Productivity', 'SMV', 'Over Time', 'Idle Time']
    values = [float(targeted_productivity), float(smv), float(over_time), float(idle_time)]
    plt.bar(categories, values, color=['blue', 'green', 'red', 'orange'])
    plt.title('Productivity Parameters - Bar')
    create_plot()

    # Scatter Plot
    plt.figure()
    plt.scatter(range(len(values)), values, color='magenta')
    plt.title('Productivity Parameters - Scatter')
    create_plot()

    # Line Plot
    plt.figure()
    plt.plot(range(len(values)), values, marker='o', color='cyan')
    plt.title('Productivity Parameters - Line')
    create_plot()

    # Pie Chart
    plt.figure()
    plt.pie(values, labels=categories, autopct='%1.1f%%')
    plt.title('Productivity Parameters - Pie')
    create_plot()

    # Histogram
    plt.figure()
    data = np.random.randn(100)
    plt.hist(data, bins=20, color='purple')
    plt.title('Random Data - Histogram')
    create_plot()

    # Boxplot
    plt.figure()
    data = np.random.rand(100, 5)
    plt.boxplot(data)
    plt.title('Random Data - Boxplot')
    create_plot()

    return render_template('submit.html', prediction_text=result_text, graphs=graphs)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
