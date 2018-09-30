#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:55:08 2018

@author: jpatrickweller
"""

from flask import Flask
app = Flask(__name__)
from bb_style import views

if __name__ == '__main__':
    app.run(debug=True)
    
#app.config.update(dict(
#        UPLOAD_FOLDER = "./website/uploads/",
#        DATA_FOLDER = "./website/models/",
#        ))