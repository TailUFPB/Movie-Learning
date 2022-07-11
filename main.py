from flask import Flask, request, render_template
from generator import Synopsis_Generator

app = Flask(__name__)
synp = Synopsis_Generator()

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/index.html")
def home():
    return render_template('index.html')

@app.route("/documentacao.html")
def documentacao():
    return render_template('documentacao.html')

@app.route("/contato.html")
def contato():
    return render_template('contato.html')

@app.route("/synp")
def test():
    return synp.get_synopsis(request.args.get('t'), request.args.get('g'), request.args.get('d'), temperature=0.7)

if __name__ == '__main__':
    app.run('0.0.0.0')