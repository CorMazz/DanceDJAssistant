# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 01:09:39 2024

@author: mazzac3


https://stackoverflow.com/questions/31948285/display-data-streamed-from-a-flask-view-as-it-updates/64974111#64974111
https://medium.com/@ruixdsgn/a-guide-to-implementing-oauth-authorization-using-spotipy-for-a-playlist-generator-app-6ab50cdf6c3
"""

import json
import base64
import spotipy
import pandas as pd
from .extensions import db
from io import BytesIO, StringIO
from .model.djassistant import DanceDJ
from .auth_controller import get_token, create_spotify_object, create_spotipy_oauth, AuthorizationError
from flask import (Blueprint, render_template, request, flash, redirect, url_for, session, current_app)


playlist_analyzer = Blueprint('playlist_analyzer', __name__)

# ----------------------------------------------------------------------------------------------------------------------
# 
# ----------------------------------------------------------------------------------------------------------------------

@playlist_analyzer.route('/select-playlist', methods=['GET', 'POST'])
def select_playlist():
    
    
    if request.method == "POST":
        if "spotify" not in (link := request.form['playlist-link']):
            flash("This must be a spotify link (looking for the word 'spotify' in the url).", "error")
        else:
            
            # Don't need scope for reading public playlists
            dj = DanceDJ(spotify_obj=spotipy.Spotify(auth_manager=create_spotipy_oauth(scope="")))
            
            session['playlist-link'] = link
            session['songs'] = dj.parse_playlist(link)
            return redirect(url_for("playlist_analyzer.view_playlist"))
        
    # Otherwise it's a get request
    return render_template('select-playlist.html')


# ----------------------------------------------------------------------------------------------------------------------
# 
# ----------------------------------------------------------------------------------------------------------------------

@playlist_analyzer.route('/view-playlist', methods=['GET', 'POST'])
def view_playlist():
    # The HTML contains logic to move from this page to the analyze playlist page upon clicking a button
    
    if (songs := session.get('songs')) is None or (link := session.get('playlist-link')) is None:
        flash("No playlist has been selected. Returning home and starting from there.", "error")
        return redirect(url_for("playlist_analyzer.select_playlist"))
    else:

        return render_template("view-playlist.html", songs=songs, link=link)
    
# ----------------------------------------------------------------------------------------------------------------------
# 
# ----------------------------------------------------------------------------------------------------------------------
    
@playlist_analyzer.route('/analyze-playlist', methods=['GET', 'POST'])
def analyze_playlist():
    
    # The HTML contains logic to move from this page to the define profile page upon clicking a button
    
    if (songs := session.get('songs')) is None or (link := session.get('playlist-link')) is None:
        flash("No playlist has been selected. Returning home and starting from there.", "error")
        return redirect(url_for("playlist_analyzer.select_playlist"))
    else:
        
        # TODO add the progress bar to the flask app
        
        # Don't need scope for analyzing songs
        dj = DanceDJ(
            spotify_obj=spotipy.Spotify(auth_manager=create_spotipy_oauth(scope="")), 
            db_session=db.session
        )
        
        processed_playlist =  dj.analyze_songs(
            songs.keys(), 
            songs.values(),
            desired_info=(
                "tempo",
                "duration",
                # "key",
                # "time_signature", 
            ),
            progress_bar=True
            )
        
        # Store the jsonified dataframe for use in future views
        session['processed-playlist'] = processed_playlist.to_json()
        
        # TODO: Add the DanceDJ().calculate_playlist_statistics() function results to this page. 
        
        render_playlist = processed_playlist.to_html(index=False)
        return render_template("view-playlist-stats.html", link=link, processed_playlist=render_playlist)
    
# ----------------------------------------------------------------------------------------------------------------------
# 
# ----------------------------------------------------------------------------------------------------------------------
    
@playlist_analyzer.route('/define-profile', methods=['GET', 'POST'])
def define_profile():
    if (processed_playlist := session.get("processed-playlist")) is None or not (n_songs := len(session.get('songs', 0))):
        flash("No playlist has been analyzed. Returning to analyze page.", "error")
        return redirect(url_for("playlist_analyzer.analyze_playlist"))
    else: 
        # We need to define a profile to read
        
        # TODO: Define a whole set of views to create a desired profile. Then let the user save and load their custom
        # profiles. 
        
        # TODO: Implement the ability to read a profile from a paint image. 
        profile_plot, fit_playlist, rearranged_playlist = None, None, None
        
        if request.method == "POST":
            
            # Don't need spotipy to create a profile or fit the playlist to the profile
            dj = DanceDJ(spotify_obj=None)
            
            profile_kwargs = dict(                
                tempo_bounds=(int(request.form.get("min_bpm", 70)), int(request.form.get("max_bpm", 130))),
                n_cycles=float(request.form.get("n_cycles", 1)), 
                horizontal_shift=float(request.form.get("horizontal_shift", 0)),
                n_songs=n_songs,
                n_points=n_songs*50,
            )
            
            target_profile = dj.generate_sinusoidal_profile(**profile_kwargs)

            session['target-profile-kwargs'] = json.dumps(profile_kwargs)
                    
            # checkbox result is either None (NoneType) or str("on")
            if (fit_playlist := request.form.get("fit_playlist")) is not None:
                analyzed_playlist = pd.read_json(StringIO(processed_playlist))
                rearranged_playlist = dj.match_tempo_profile(
                    target_profile,
                    analyzed_playlist,
                    method="euclidean"  # no need to upsample, profile is already oversampled
                )
                
                # Store the jsonified dataframe for use in future views
                session['rearranged-playlist'] = rearranged_playlist.to_json()
            
            fig, ax = dj.plot_profile_matplotlib(
                target_profile=target_profile, 
                analyzed_playlist=rearranged_playlist, 
                target_profile_kwargs=dict(markersize=4),
                fig_kwargs=dict(figsize=(8,6)),
            )
            
            # Convert the image to bytes so I can ship it off to the HTML and render it
            figfile = BytesIO()
            fig.savefig(figfile, format='png')
            figfile.seek(0)  # rewind to beginning of file

            profile_plot = base64.b64encode(figfile.getvalue()).decode('utf8')
            
        # Run this regardless if it's a get request or a post request
        return_value = render_template(
            "define-profile.html", 
            form_data=request.form, 
            profile_plot=profile_plot, 
            continue_button_active= True if fit_playlist is not None else False
        )
        return return_value
        
# ----------------------------------------------------------------------------------------------------------------------
# 
# ----------------------------------------------------------------------------------------------------------------------
    
@playlist_analyzer.route('/upload-playlist', methods=['POST', 'GET'])
def upload_playlist():
    if (
            (rearranged_playlist := session.get("rearranged-playlist")) is None 
            or (target_profile_kwargs := session.get("target-profile-kwargs")) is None 
            or not (len(session.get('songs', 0)))
            
        ):
        flash("No playlist has been rearranged. Returning to define profile page.", "error")
        return redirect(url_for("playlist_analyzer.define_profile"))
    
    # Don't need spotipy to create a profile or fit the playlist to the profile
    dj = DanceDJ(spotify_obj=None)
    
    # convert the rearranged playlist and target profile back from json
    rearranged_playlist = pd.read_json(StringIO(rearranged_playlist))
    target_profile_kwargs = json.loads(target_profile_kwargs)
    target_profile = dj.generate_sinusoidal_profile(**target_profile_kwargs)
    
    fig, ax = dj.plot_profile_matplotlib(target_profile=target_profile, analyzed_playlist=rearranged_playlist)
    
    # Convert the image to bytes so I can ship it off to the HTML and render it
    figfile = BytesIO()
    fig.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file

    profile_plot = base64.b64encode(figfile.getvalue()).decode('utf8')
    
    if request.method == "POST":  # The name must be filled out to submit a post request
        
        try:
            dj = DanceDJ(spotify_obj=create_spotify_object(session))
        except AuthorizationError:
            print("Not authorized, redirecting")
            return redirect(url_for('auth.grant_spotify_access'))
        
        flash("Authorized, attempting to upload.", "success")
        
        playlist_url = dj.save_playlist(request.form.get("playlist_name"), list(rearranged_playlist.index),) #cover_image_b64=fig)
        
        return render_template('wait-for-playlist.html', playlist_url=playlist_url)
    

    return render_template("upload-playlist.html", form_data=request.form, profile_plot=profile_plot)
    

    
    