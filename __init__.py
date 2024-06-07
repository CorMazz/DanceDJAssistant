import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from pathlib import Path

db = SQLAlchemy()
db_name = "database.db"

########################################################################################################################
# 
########################################################################################################################

def create_app(test_config=None):
    """The app factory function"""
    app = Flask(__name__, instance_relative_config=True)
    
    app.config.from_mapping(
        SECRET_KEY="dev",
        SQLALCHEMY_DATABASE_URI=f'sqlite:///{db_name}'
    )
    
    db.init_app(app)
    # Load configuration depending on if I'm testing or just running
    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)
        
    # ensure the instance folder exists
    os.makedirs(app.instance_path, exist_ok=True)
    
    print(f"{app.instance_path=}")
    
    with app.app_context():
        create_database(app)
    
    @app.route('/dev')
    def dev():
        return "<h1>This is a development test page. Guess the website is up :)</h1>"
    
    return app

########################################################################################################################
#
########################################################################################################################

def create_database(app: Flask, db_name: str):
    db_path = os.path.join(app.instance_path, db_name)
    if not os.path.exists(db_path):
        db.create_all()
        print("Created App Database")

if __name__ == "__main__":
    app = create_app()
    app.run(debug=False)