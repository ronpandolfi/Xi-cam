import jinja2
import os
import yaml
import glob

from flask import Flask, render_template
app = Flask(__name__)

def getAttributions():
    attributions = []
    for path in glob.glob('../yaml/attributions/*'):
        with open(path,'r') as f:
            attributions.append(yaml.load(f.read()))
    return attributions


@app.route('/')
def generateMOTD():
    return render_template('template.html',attributions=getAttributions())

if __name__ == '__main__':
    app.run(debug=True)