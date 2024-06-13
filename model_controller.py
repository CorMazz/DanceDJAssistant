# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 01:09:39 2024

@author: mazzac3


"""

import os
import base64
from io import BytesIO
from model.DJAssistant import DanceDJ
from flask import (Blueprint, render_template, request, flash, redirect, url_for, session)

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
            os.environ["SPOTIPY_CLIENT_ID"] = "b6eaa41c44d44f919fc2f49cba43767a"
            os.environ["SPOTIPY_CLIENT_SECRET"] = "7b39402661b44941b2d8a2b1209a3797"
            os.environ["SPOTIPY_REDIRECT_URI"] = "http://localhost:8080"
            session['playlist-link'] = link
            session['songs'] = DanceDJ().parse_playlist(link)
            return redirect(url_for("playlist_analyzer.view_playlist"))
        
    # Otherwise it's a get request. 
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
        
        # TODO: Refactor this so that the DanceDJ class has the caching functionality with SQL, it just accepts the DB
        # connection to use from the Flask app
        processed_playlist =  DanceDJ().analyze_songs(
            songs.keys(), 
            songs.values(),
            desired_info=(
                "tempo",
                "key",
                "time_signature", 
            ),
            progress_bar=True
            )
        
        # Store the jsonified dataframe for use in future views
        session['processed-playlist'] = processed_playlist.to_json()
        print(f"{session['processed-playlist']=}")
        
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
        profile_plot = None
        
        if request.method == "POST":
            # Read the items defined in the post request
            if any((not request.form.get(key, False) for key in ["min_bpm", "max_bpm", "n_cycles"])):
                flash("Values must be non-zero, positive floats/ints.", "error")
            else:
                profile = DanceDJ().generate_sinusoidal_profile(
                    (int(request.form.get("min_bpm")), int(request.form.get("max_bpm"))),
                    float(request.form.get("n_cycles")), 
                    float(request.form.get("horizontal_shift", 0)),
                    n_songs,
                )
                
                fig, ax = DanceDJ().plot_profile(profile)
                
                # Convert the image to bytes so I can ship it off to the HTML and render it
                figfile = BytesIO()
                fig.savefig(figfile, format='png')
                figfile.seek(0)  # rewind to beginning of file

                profile_plot = base64.b64encode(figfile.getvalue()).decode('utf8')
            
        # Run this regardless if it's a get request or a post request
        return render_template("define-profile.html", form_data=request.form, profile_plot=profile_plot)
        