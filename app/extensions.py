# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:48:33 2024

@author: mazzac3
"""

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import declarative_base

Base = declarative_base()
db = SQLAlchemy(model_class=Base)
db_name = "database.db"