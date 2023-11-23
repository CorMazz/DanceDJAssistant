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
    os.environ["SPOTIPY_REDIRECT_URI"] = "http://localhost"
    
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
        playlist_id="https://open.spotify.com/playlist/5i5aRhCzljuixCrSES3pYH?si=ecc1f0a28c344516" # Sarah
        #"https://open.spotify.com/playlist/601jBGXOfsMFZh3DfNcrVh?si=af574fa0ea364c27" # Mine
    )
    
    # Truncate down to n songs
    playlist_songs = dict(list(playlist_songs.items())[:1000])

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
    
# -------------------------------------------------------------------------------------------------
# Match Profile 
# -------------------------------------------------------------------------------------------------       

    # Define a target tempo profile for n songs
    n_songs = 50
    n_cycles = 2
    amplitude = 22
    mean = 92
    song_idx = np.linspace(0, n_cycles*2*np.pi, n_songs)
    song_profile = amplitude*np.sin(song_idx) + mean
    
    # Match the profile with songs
    selected_songs = dj.match_tempo_profile(song_profile, playlist_summary)
    
    # Quantify the error
    error = mean_absolute_error(song_profile, selected_songs['tempo'])
    
    # Plot the target profile
    if 1:
        fig, ax = plt.subplots()
        ax.plot(song_profile, 'k--', label="Target Profile")
        ax.plot(selected_songs['tempo'].to_numpy(), label='Achieved Profile')
        ax.set_ylabel("Tempo")
        ax.set_xlabel("Song Number")
        ax.set_title(f"Target Song Profile vs. Achieved Profile\n MAE = {error:.2f} BPM")
        ax.legend()
        fig.show()
        
        
# -------------------------------------------------------------------------------------------------
# Save the Playlist
# -------------------------------------------------------------------------------------------------   
        
    if input("Save playlist? (y/n)").lower() != "y":
        raise KeyboardInterrupt("Program terminated by user.")
        
    
    dj.save_playlist(
        (f"{n_songs} Songs -- {n_cycles} Cycles within ({mean - amplitude}, {mean+amplitude}) BPM -- "
         f"MAE = {error:.2f} BPM")
        [:100],
        list(selected_songs.index)
    )
    
    