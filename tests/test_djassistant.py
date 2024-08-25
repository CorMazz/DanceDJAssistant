# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:53:12 2024

@author: mazzac3
"""



import os
import sys
import spotipy
import numpy as np
import plotly.express as px

sys.path.append(r"C:\Users\mazzac3\Documents\GitHub\DanceDJAssistant")
from djassistant import DanceDJ
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from contextlib import contextmanager

def load_dev_env_variables(env_file=".env"):
    """
    Load environment variables from a .env file if it exists. This is for easier development. My dev environment will 
    have the .env file available, but the docker container will not have the file. 

    Parameters
    ----------
    env_file : str, optional
        The path to the .env file (default is ".env").

    Returns
    -------
    None
    """
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                # Strip whitespace and ignore comments and empty lines
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Split the line into key and value
                key, value = line.split('=', 1)

                # Remove potential leading/trailing whitespace and enclosing quotes
                key = key.strip()
                value = value.strip().strip('\'"')

                # Set the environment variable
                os.environ[key] = value

        print(f"Loaded environment variables from {env_file}")
    else:
        print(f"No .env file found at {env_file}")



########################################################################################################################
########################################################################################################################
# Main
########################################################################################################################
########################################################################################################################

# Playlist must have at least 20 songs.
pl_url = "https://open.spotify.com/playlist/7MVcmq2Dvir4lRvIWQ4J0f?si=nc-knGZrTUe4sXAuwgnsMw&pi=G1Ml4W56TOize" 


if __name__ == "__main__":
    
    test_db = 1
    test_plot_plotly = 1
    
    load_dev_env_variables(r"C:\Users\mazzac3\Documents\GitHub\dance-dj-webapp\.env")
    
    auth_manager = spotipy.oauth2.SpotifyOAuth(
        client_id=os.environ["SPOTIPY_CLIENT_ID"],
        client_secret=os.environ["SPOTIPY_CLIENT_SECRET"],
        redirect_uri="http://localhost:8080",
    )
    
    spotify_obj = spotipy.Spotify(auth_manager=auth_manager)
    
    
    if test_db:
    
        # Define your database connection URL
        DATABASE_URL = "sqlite:///test_only.db"
        
        # Create the SQLAlchemy engine
        engine = create_engine(DATABASE_URL)
        
        # Clean the database so that each test is fresh
        meta = MetaData()
        meta.reflect(bind=engine)
        meta.drop_all(bind=engine)
            
        ################################################################################################################
        #%% Test __init__, parse_playlist, and analyze_songs with/without database functionality
        ################################################################################################################
    
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
                
                # Make sure this return_api_response works so set it to true
                playlist_object= dj.parse_playlist(playlist_id=playlist, return_api_response=True)[1]
                
                # Truncate down to n songs
                playlist_songs = dict(list(playlist_songs.items())[:n_songs])
            
                # Analyze the playlist
                playlist_summary = dj.analyze_songs(
                    song_ids=playlist_songs.keys(),
                    song_names=playlist_songs.values(),
                )
                
    ####################################################################################################################
    #%% Plotly Functionality
    ####################################################################################################################
    
    if test_plot_plotly:
        
        # Define your database connection URL
        DATABASE_URL = "sqlite:///test_only.db"
        
        # Create the SQLAlchemy engine
        engine = create_engine(DATABASE_URL)
        
        with Session(engine) as session:
        
            dj=DanceDJ(spotify_obj=spotify_obj, db_session=session)

            # Parse the given playlist
            playlist_songs = dj.parse_playlist(playlist_id=pl_url)
            
            # Analyze the playlist
            playlist_summary = dj.analyze_songs(
                song_ids=playlist_songs.keys(),
                song_names=playlist_songs.values(),
            )
            
            n_songs = len(playlist_summary)
            target_profile = dj.generate_sinusoidal_profile((70, 130), 6, 0, n_songs, 0.1, n_songs * 5)
            
            fig = dj.plot_playlist_plotly(analyzed_playlist=playlist_summary, target_profile=target_profile)
            dj.plot_profile_matplotlib(target_profile=target_profile, analyzed_playlist=playlist_summary)
            fig.write_html("./unordered_playlist.html")
            
            rearranged_playlist = dj.match_tempo_profile(target_profile[:, 1], playlist_summary, method="euclidean")
            fig = dj.plot_playlist_plotly(analyzed_playlist=rearranged_playlist, target_profile=target_profile)
            dj.plot_profile_matplotlib(target_profile=target_profile, analyzed_playlist=rearranged_playlist)
            fig.write_html("./ordered_playlist.html")
            
        # fig.show(renderer="browser")

        
        # fig = dj.plot_
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

