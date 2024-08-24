import sys
import os

# Add the parent directory (Graphs) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, send_file
import io
import base64
from resultAnalysis import importFiles, makeGraph
import matplotlib.pyplot as plt


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        models = request.form.getlist('models')
        num_movies = list(map(int, request.form.getlist('num_movies')))
        gpt_temp = float(request.form.get('gpt_temp'))
        ollama_temp = list(map(float, request.form.getlist('ollama_temp')))
        attributes = request.form.getlist('attributes')
        attributsForDistinctionInColor = request.form.getlist('attributsForDistinctionInColor')
        nameForDistinctionInColor = request.form.get('nameForDistinctionInColor')
        width = float(request.form.get('plotWidth'))
        height = float(request.form.get('plotheight'))

        
        filtered_models = importFiles("./results/", models, num_movies, gpt_temp, None, ollama_temp, None)

        if filtered_models:
            figs = []
            for attr in attributes:
                fig, ax = plt.subplots(figsize=(width, height))
                makeGraph("Model", attr, filtered_models, ['blue', 'green', 'red', 'yellow', 'purple'], 999999, attributsForDistinctionInColor, nameForDistinctionInColor,ax)
                
                img = io.BytesIO()
                fig.savefig(img, format='png')
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode()
                figs.append(plot_url)
                plt.close(fig)

            return render_template('results.html', plots=figs)
            
    filtered_models = importFiles("./results/")
    modelDict = {model.model() for model in filtered_models}
    sorted_models = sorted(list(modelDict), key=str.lower)
    return render_template('index.html',models=sorted_models)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)