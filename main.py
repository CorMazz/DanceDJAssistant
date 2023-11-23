# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:52:17 2023

@author: mazzac3
"""
import io
import os
import base64
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
        playlist_id=
        # "https://open.spotify.com/playlist/0m66VKTjiV89OJsE0NHRFa?si=ca4cd434f9a1437a" # Liz 11/28
        "https://open.spotify.com/playlist/5i5aRhCzljuixCrSES3pYH?si=ecc1f0a28c344516" # Sarah
        #"https://open.spotify.com/playlist/601jBGXOfsMFZh3DfNcrVh?si=af574fa0ea364c27" # Mine
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
    
# -------------------------------------------------------------------------------------------------
# Match Profile 
# -------------------------------------------------------------------------------------------------       

#     # Define a target tempo profile for n songs
#     n_songs = len(playlist_summary)
#     n_cycles = 1
#     amplitude = 27
#     mean = 100
#     song_idx = np.linspace(0, n_cycles*2*np.pi, n_songs)
#     song_profile = amplitude*np.sin(song_idx) + mean
    
#     # Match the profile with songs
#     selected_songs = dj.match_tempo_profile(
#         song_profile, 
#         playlist_summary,
#         method="supersampled_euclidean"
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
#     ax.set_ylabel("Tempo")
#     ax.set_xlabel("Song Number")
#     ax.set_title(f"Target Song Profile vs. Achieved Profile\n MAE = {error:.2f} BPM")
#     ax.legend()

#     # Convert the target profile image to a b64 byte string
#     cover_image_b64: bytes | None = None
#     with io.BytesIO() as buffer:
#         fig.savefig(buffer, format='jpeg')
        
#         # Check if the buffer size is within the limit (256 KB)
#         if buffer.tell() <= 256 * 1024:
#             print("\n\nSetting cover image")
#             buffer.seek(0)  # Reset buffer position to start
#             cover_image_b64 = base64.b64encode(buffer.getvalue())
        
            
# # -------------------------------------------------------------------------------------------------
# # Save the Playlist
# # -------------------------------------------------------------------------------------------------   
        
#     if input("Save playlist? (y/n)").lower() != "y":
#         raise KeyboardInterrupt("Program terminated by user.")
        
#     # Set the playlist name
#     playlist_name = (
#         f"Liz's 11/28 Playlist Rearranged -- {n_cycles} Cycle{'s' if n_cycles != 1 else ''} -- ({mean - amplitude}"
#         f", {mean+amplitude}) BPM -- MAE = {error:.2f} BPM"
#     )
    
#     # Save the playlist
#     dj.save_playlist(
#         playlist_name=playlist_name,
#         playlist_songs=list(selected_songs.index),
#         cover_image_b64=cover_image_b64
#     )
    
    