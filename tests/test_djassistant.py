# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:53:12 2024

@author: mazzac3
"""



import os
import sys
import spotipy
import numpy as np

sys.path.append(r"C:\Users\mazzac3\Documents\GitHub\DanceDJAssistant\app\model")
from djassistant import DanceDJ
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from contextlib import contextmanager




########################################################################################################################
########################################################################################################################
# Main
########################################################################################################################
########################################################################################################################

# Playlist must have at least 20 songs.
pl_url = "https://open.spotify.com/playlist/7MVcmq2Dvir4lRvIWQ4J0f?si=nc-knGZrTUe4sXAuwgnsMw&pi=G1Ml4W56TOize" 


if __name__ == "__main__":
    
    auth_manager = spotipy.oauth2.SpotifyOAuth(
        client_id="b6eaa41c44d44f919fc2f49cba43767a",
        client_secret="7b39402661b44941b2d8a2b1209a3797",
        redirect_uri="http://localhost:8080",
    )
    
    spotify_obj = spotipy.Spotify(auth_manager=auth_manager)
    
    # Define your database connection URL
    DATABASE_URL = "sqlite:///test_only.db"
    
    # Create the SQLAlchemy engine
    engine = create_engine(DATABASE_URL)
    
    # Clean the database so that each test is fresh
    meta = MetaData()
    meta.reflect(bind=engine)
    meta.drop_all(bind=engine)
        
    ####################################################################################################################
    #%% Test __init__, parse_playlist, and analyze_songs with/without database functionality
    ####################################################################################################################

    with Session(engine) as session:
    
        test_config_settings = {
            # Each time the test code runs, the database gets changed slightly. 
            
            
            "No Database": dict(
                dj_obj=DanceDJ(spotify_obj=spotify_obj),
                playlist=pl_url,
                n_songs=20,
            ),
      
            "Blank Database": dict(
                dj_obj=DanceDJ(spotify_obj=spotify_obj, db_session=session),
                playlist=pl_url,
                n_songs=10,
            ),
            
            "Half Populated Database": dict(
                dj_obj=DanceDJ(spotify_obj=spotify_obj, db_session=session),
                playlist=pl_url,
                n_songs=20,
            ),
            
            "Fully Populated Database": dict(
                dj_obj=DanceDJ(spotify_obj=spotify_obj, db_session=session),
                playlist=pl_url,
                n_songs=20,
            ),
            
        }
    
        
        for test_name, test_config in test_config_settings.items():
            
            dj, playlist, n_songs = test_config.values()
    
            # -------------------------------------------------------------------------------------------------
            # Process the Playlist
            # -------------------------------------------------------------------------------------------------       
            
        
            # Parse the given playlist
            playlist_songs = dj.parse_playlist(playlist_id=playlist)
            
            # Truncate down to n songs
            playlist_songs = dict(list(playlist_songs.items())[:n_songs])
        
            # Analyze the playlist
            playlist_summary = dj.analyze_songs(
                song_ids=playlist_songs.keys(),
                song_names=playlist_songs.values(),
            )
            
        
            
# # -------------------------------------------------------------------------------------------------
# # Match Profile 
# # -------------------------------------------------------------------------------------------------       

#     # Define a target tempo profile for n songs
#     n_songs = len(playlist_summary)
#     n_cycles = 5
#     amplitude = 23
#     mean = 97
#     song_idx = np.linspace(0, n_cycles*2*np.pi, n_songs)
#     song_profile = amplitude*np.sin(song_idx + 5) + mean
    
#     # Match the profile with songs
#     selected_songs = dj.match_tempo_profile(
#         song_profile, 
#         playlist_summary,
#         method="upsampled_euclidean"
#     )
    
#     if isinstance(selected_songs, dict):
#         song_profile = song_profile[selected_songs['profile_indices']]
#         selected_songs = selected_songs['selected_songs']
        
#     # Quantify the error
#     error = mean_absolute_error(song_profile, selected_songs['tempo'])
    
#     # Plot the target profile
#     fig, ax = plt.subplots(figsize=(5,5))
#     ax.plot(song_profile, 'k--', marker="o", label="Target Profile")
#     ax.plot(selected_songs['tempo'].to_numpy(), marker="*", label='Achieved Profile')
#     ax.set_ylabel("Tempo (BPM)")
#     ax.set_xlabel("Song Number")
#     ax.set_title(f"Target Song Profile vs. Achieved Profile\n MAE = {error:.2f} BPM")
#     ax.legend()

    
# -------------------------------------------------------------------------------------------------
# Save the Playlist
# -------------------------------------------------------------------------------------------------   

# if input("Save playlist? (y/n)").lower() != "y":
#     raise KeyboardInterrupt("Program terminated by user.")

# # Set the playlist name
# playlist_name = (
#     f"5/14 -- {n_cycles} Cycle{'s' if n_cycles != 1 else ''} -- ({mean - amplitude}"
#     f", {mean+amplitude}) BPM -- MAE = {error:.2f} BPM"
# )

# # Save the playlist
# dj.save_playlist(
#     playlist_name=playlist_name,
#     playlist_songs=list(selected_songs.index),
#     cover_image_b64=fig,
# )

