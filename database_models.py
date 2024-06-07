# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 06:33:03 2024

@author: mazzac3

Create the SQLAlchemy models used to store the data
"""

from . import db
from flask_login import UserMixin


class Song(db.Model):
    """Store songs in the database. Each song can be identiied by its URL/I and it's BPM."""
    url = db.Column(db.String(300), primary_key=True)
    bpm = db.Column(db.Integer)
        

    
# class User(db.Model, UserMixin):
#     """Copied from https://www.youtube.com/watch?v=dam0GPOAvVI for future reference. Unused"""
#     id = db.Column(db.Integer, primary_key=True)
#     email = db.Column(db.String(150), unique=True)
#     password = db.Column(db.String(150))
#     first_name = db.Column(db.String(150))

