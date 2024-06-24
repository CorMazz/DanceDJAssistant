import os
from pathlib import Path
from .extensions import db, db_name
from .model_controller import playlist_analyzer
from .auth_controller import auth
from flask import Flask, render_template

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
        SPOTIPY_CLIENT_ID="b6eaa41c44d44f919fc2f49cba43767a",
        SPOTIPY_CLIENT_SECRET= "7b39402661b44941b2d8a2b1209a3797",  # Refresh me before making public
        SPOTIPY_REDIRECT_URI= "http://127.0.0.1:5000/spotify-api-callback",
        
    )
    
    # Load configuration depending on if I'm testing or just running
    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)
    
    # ensure the instance folder exists
    os.makedirs(app.instance_path, exist_ok=True)
    
    with app.app_context():
        create_database(app)
    
    # Start the SQL database
    db.init_app(app)

    
    @app.route('/dev')
    def dev():
        return render_template("base.html")
    

    app.register_blueprint(playlist_analyzer)
    app.register_blueprint(auth)
    
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