# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:52:17 2023

@author: mazzac3
"""


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Memory
from DJAssistant import DanceDJ
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

###################################################################################################
# Main
###################################################################################################

if __name__ == "__main__":
    
    os.environ["SPOTIPY_CLIENT_ID"] = "b6eaa41c44d44f919fc2f49cba43767a"
    os.environ["SPOTIPY_CLIENT_SECRET"] = "7b39402661b44941b2d8a2b1209a3797"
    os.environ["SPOTIPY_REDIRECT_URI"] = "http://localhost:8080"
    
    # Create a DanceDJ object
    dj = DanceDJ()
    
    # Define a caching directory
    cachedir = "./cache"
    memory = Memory(cachedir, verbose=0)
    
    # Wrap the analyze songs method and the parse_playlist method
    @memory.cache
    def analyze_songs(song_id, song_name):
        return dj.analyze_songs([song_id], [song_name], progress_bar=False)
        
    
    @memory.cache
    def parse_playlist(playlist_id):
        return dj.parse_playlist(playlist_id)
    
    
# -------------------------------------------------------------------------------------------------
# Process the Playlist
# -------------------------------------------------------------------------------------------------       

    
    # Parse the given playlist
    playlist_songs = parse_playlist(
        playlist_id=
        # "https://open.spotify.com/playlist/0m66VKTjiV89OJsE0NHRFa?si=ca4cd434f9a1437a" # Liz 11/28
        # "https://open.spotify.com/playlist/5i5aRhCzljuixCrSES3pYH?si=ecc1f0a28c344516" # Sarah
        #"https://open.spotify.com/playlist/601jBGXOfsMFZh3DfNcrVh?si=af574fa0ea364c27" # Mine
        # "https://open.spotify.com/playlist/7DFID0tvfRns1xbMgpsVIO?si=c3e587f750c64a35" # Liz 4/2
        # "https://open.spotify.com/playlist/0nkHtrospRSZaxzdWP08iZ" # Cory's Copy of Liz 4/2
        "https://open.spotify.com/playlist/7MVcmq2Dvir4lRvIWQ4J0f?si=nc-knGZrTUe4sXAuwgnsMw&pi=G1Ml4W56TOize"  # Cory's Copy of Liz 5/14
    )
    
    # Truncate down to n songs
    # playlist_songs = dict(list(playlist_songs.items())[:20])

    # Analyze the playlist
    playlist_summary = pd.concat(
        [
            analyze_songs(
                song_id=song_id,
                song_name=song_name,
                ) 
            for song_id, song_name in tqdm(playlist_songs.items(), desc="Analyzing Songs")
        ]
    )
    
    # Adjust the tempo (out here because I don't want to re-analyze all those songs since spotipy is timing out)
    playlist_summary = dj.adjust_tempo(playlist_summary, (60, 130))
    
# -------------------------------------------------------------------------------------------------
# Match Profile 
# -------------------------------------------------------------------------------------------------       

    # Define a target tempo profile for n songs
    n_songs = len(playlist_summary)
    n_cycles = 5
    amplitude = 23
    mean = 97
    song_idx = np.linspace(0, n_cycles*2*np.pi, n_songs)
    song_profile = amplitude*np.sin(song_idx + 5) + mean
    
    # Match the profile with songs
    selected_songs = dj.match_tempo_profile(
        song_profile, 
        playlist_summary,
        method="upsampled_euclidean"
    )
    
    if isinstance(selected_songs, dict):
        song_profile = song_profile[selected_songs['profile_indices']]
        selected_songs = selected_songs['selected_songs']
        
    # Quantify the error
    error = mean_absolute_error(song_profile, selected_songs['tempo'])
    
    # Plot the target profile
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(song_profile, 'k--', marker="o", label="Target Profile")
    ax.plot(selected_songs['tempo'].to_numpy(), marker="*", label='Achieved Profile')
    ax.set_ylabel("Tempo (BPM)")
    ax.set_xlabel("Song Number")
    ax.set_title(f"Target Song Profile vs. Achieved Profile\n MAE = {error:.2f} BPM")
    ax.legend()

            
# -------------------------------------------------------------------------------------------------
# Save the Playlist
# -------------------------------------------------------------------------------------------------   
        
    if input("Save playlist? (y/n)").lower() != "y":
        raise KeyboardInterrupt("Program terminated by user.")
        
    # Set the playlist name
    playlist_name = (
        f"5/14 -- {n_cycles} Cycle{'s' if n_cycles != 1 else ''} -- ({mean - amplitude}"
        f", {mean+amplitude}) BPM -- MAE = {error:.2f} BPM"
    )
    
    # Save the playlist
    dj.save_playlist(
        playlist_name=playlist_name,
        playlist_songs=list(selected_songs.index),
        cover_image_b64=fig,
    )
    
    