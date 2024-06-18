import os
from flask import Flask, render_template
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
        SQLALCHEMY_DATABASE_URI=f'sqlite:///{db_name}',
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
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
        return render_template("base.html")
    
    # Register pages
    from model_controller import playlist_analyzer
    app.register_blueprint(playlist_analyzer)
    
    return app

########################################################################################################################
#
########################################################################################################################

def create_database(app: Flask):
    db_path = os.path.join(app.instance_path, db_name)
    if not os.path.exists(db_path):
        db.create_all()
        print("Created App Database")

if __name__ == "__main__":
    app = create_app()
    app.run(debug=False)