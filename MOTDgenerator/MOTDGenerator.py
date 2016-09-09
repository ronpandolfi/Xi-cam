import jinja2
import os
import yaml
import glob

from flask import Flask, render_template
from os import path
app = Flask(__name__)

def getAttributions():
    attributions = []
    toplevel = []
    for p in glob.glob('../yaml/attributions/*'):
        with open(p,'r') as f:
            attrDict=yaml.load(f.read())
            attrDict['codename']=path.splitext(path.basename(p))[0]
            if 'toplevel' in attrDict:
                toplevel.append(attrDict)
            else:
                attributions.append(attrDict)



    return toplevel+attributions


@app.route('/')
def generateMOTD():
    return render_template('template.html',attributions=getAttributions())

if __name__ == '__main__':
    app.run(debug=True)