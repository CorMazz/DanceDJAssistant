# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 04:46:15 2024

@author: mazzac3


Contains functions to handle the database for the DJAssistant. The database will store song information,
"""

import sqlite3
import click
from flask import current_app, g

def get_db():
    
    # Make a new DB connection if we don't have an existing one
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        
        g.db.row_factory = sqlite3.Row
        
    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    
    if db is not None:
        db.close()
        
    