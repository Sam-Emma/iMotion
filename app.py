# -*- coding: utf-8 -*-

from flask import Flask, request,  render_template
import model as m
app = Flask(__name__)

@app.route("/", methods =["POST","GET"])
def Home():
    text  =""
    predict_real =""
    confid =""
    if request.method == "POST":
        emo = request.form["emo"]
        text, predict_real, confid = m.predict_emotion(emo)
        
    return render_template("index.html", message = text, emotion = predict_real, confidence = confid)
#@app.route('/sub', methods = ['POST'])
#def submit():
#    if request.method == "POST":
#        name = request.form["username"]
#        return render_template("sub.html", n =name)

if __name__ == "__main__":
    app.run(debug = True,use_reloader=False)