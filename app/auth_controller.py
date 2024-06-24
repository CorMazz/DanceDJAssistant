# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:24:03 2024

@author: mazzac3

https://stackoverflow.com/questions/57580411/storing-spotify-token-in-flask-session-using-spotipy
"""

import time
import spotipy
from flask import (Blueprint, render_template, request, flash, redirect, url_for, session, jsonify, current_app)



auth = Blueprint('auth', __name__,)

########################################################################################################################
########################################################################################################################
# Views
########################################################################################################################
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# 
# ----------------------------------------------------------------------------------------------------------------------

# authorization-code-flow Step 1. Have your application request authorization; 
# the user logs in and authorizes access
@auth.route("/grant-spotify-access")
def grant_spotify_access():
    """https://stackoverflow.com/questions/57580411/storing-spotify-token-in-flask-session-using-spotipy"""
    
    # Don't reuse a SpotifyOAuth object because they store token info and you could leak user tokens if you reuse a SpotifyOAuth object
    sp_oauth = create_spotipy_oauth()
    
    auth_url = sp_oauth.get_authorize_url()
    print(f"{auth_url=}")
    return redirect(auth_url)

# ----------------------------------------------------------------------------------------------------------------------
# 
# ----------------------------------------------------------------------------------------------------------------------

# authorization-code-flow Step 2.
# Have your application request refresh and access tokens;
# Spotify returns access and refresh tokens
@auth.route("/spotify-api-callback")
def spotify_api_callback():
    # Don't reuse a SpotifyOAuth object because they store token info and you could leak user tokens if you reuse a SpotifyOAuth object
    sp_oauth = create_spotipy_oauth()

    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)

    # Saving the access token along with all other token related info
    session["token_info"] = token_info

    return redirect(url_for('playlist_analyzer.upload_playlist'))



########################################################################################################################
########################################################################################################################
# Utility Functions
########################################################################################################################
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# Create Spotify Object
# ----------------------------------------------------------------------------------------------------------------------

def create_spotify_object(session) -> spotipy.Spotify:
    """Checks if a token has been saved in the session. If there is one, uses it to create and return a Spotify object.
    If not, raises an AuthorizationError.
    """
    session['token_info'], authorized = get_token(session)
    session.modified = True
    if not authorized:
        raise AuthorizationError("The Spotify session is not authenticated. Restart the authorization flow.")
    
    return spotipy.Spotify(auth=session.get('token_info').get('access_token'))

# ----------------------------------------------------------------------------------------------------------------------
# Get Token
# ----------------------------------------------------------------------------------------------------------------------

# Checks to see if token is valid and gets a new token if not
def get_token(session):
    token_valid = False
    token_info = session.get("token_info", {})

    # Checking if the session already has a token stored
    if not (session.get('token_info', False)):
        token_valid = False
        return token_info, token_valid

    # Checking if token has expired
    now = int(time.time())
    is_token_expired = session.get('token_info').get('expires_at') - now < 60

    # Refreshing token if it has expired
    if is_token_expired:
        # Don't reuse a SpotifyOAuth object because they store token info and you could leak user tokens if you reuse a SpotifyOAuth object
        sp_oauth = create_spotipy_oauth()

        token_info = sp_oauth.refresh_access_token(session.get('token_info').get('refresh_token'))

    token_valid = True
    return token_info, token_valid


# ----------------------------------------------------------------------------------------------------------------------
# Create Spotipy OAuth Object
# ----------------------------------------------------------------------------------------------------------------------

def create_spotipy_oauth(
        scope: str = "playlist-modify-public playlist-modify-private ugc-image-upload"
    ) -> spotipy.oauth2.SpotifyOAuth:
    
    sp_oauth = spotipy.oauth2.SpotifyOAuth(
        scope=scope,
        **dict(
            client_id=current_app.config['SPOTIPY_CLIENT_ID'],
            client_secret=current_app.config["SPOTIPY_CLIENT_SECRET"],
            redirect_uri=current_app.config["SPOTIPY_REDIRECT_URI"]
        ),
    )
    return sp_oauth

class AuthorizationError(Exception):
    """A utility error to indicate that the Spotify session is not logged in."""
    pass