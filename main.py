from flask import Flask, redirect, url_for, render_template

app = Flask(__name__)


@app.route("/")
def html():
    return render_template("index.html")

@app.route("/playground")
def playground():
    return render_template("playground.html")

@app.route("/manual")
def manual():
    return render_template("manual.html")






if __name__ == "__main__":
    app.run(debug=True)