# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:28:03 2023

@author: mazzac3
"""

import spotipy
import numpy as np
import pandas as pd
from tqdm import tqdm
from spotipy.oauth2 import SpotifyOAuth
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment



class DanceDJ:
    
    def __init__(self, scope="playlist-modify-public playlist-modify-private ugc-image-upload"):
        """
        Instantiates a connection to the Spotify API. See the Spotipy documentation to learn 
        about the authorization scope.
        """
        
        self._sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))
        
###################################################################################################
# Primary Methods
###################################################################################################

# -------------------------------------------------------------------------------------------------
# Parse Playlist
# -------------------------------------------------------------------------------------------------

    def parse_playlist(self, playlist_id: str, max_tracks: int = 5e3):
        """
        Parse a given playlist and grab the IDs and names of it's songs.
    
        Parameters
        ----------
        playlist_id : str
            The Spotify ID of a playlist. Can be a url. 
    
        Returns
        -------
        parsed_songs : dict
            A dictionary where the key is the url for a song, and the value is the song name. 
            This was on purpose, since urls are unique, as keys need to be. 
    
        """
        
        # Get the total number of tracks in the playlist
        playlist = self._sp.playlist(playlist_id)
        total_tracks = playlist['tracks']['total']
        
        # Also grab the first 100 tracks since we already called the API
        
        # Initialize storage for the track information
        parsed_songs = {song['track']['external_urls']['spotify']: song['track']['name']
                        for song in playlist['tracks']['items']}
        
        # Initialize an offset variable to paginate through tracks
        offset, limit = 100, 100  # Spotify API limits to 100 tracks per request
        
        # Loop until all tracks are retrieved
        while offset < total_tracks and offset < max_tracks:
            # Get a batch of tracks using offset and limit
            tracks = self._sp.playlist_tracks(playlist_id, offset=offset, limit=limit)['items']
    
            # Add the desired information from those songs
            parsed_songs.update(
                {song['track']['external_urls']['spotify']: song['track']['name']
                 for song in tracks}
            )
    
            # Increment the offset for the next batch
            offset += limit
            
        return parsed_songs
    
# -------------------------------------------------------------------------------------------------
# Analyze Songs
# -------------------------------------------------------------------------------------------------

    def analyze_songs(
        self, 
        song_ids: list, 
        song_names: list | None = None,
        adjust_tempo: tuple[int, int] | None = (60, 140),
        progress_bar: bool = True
        ) -> pd.DataFrame:
        """
        Given a list of song_ids, analyzes the songs and returns a DataFrame with important 
        information analyzed from the songs.

        Parameters
        ----------
        song_ids : list
            A list of the song IDs. Can be URLs.
        song_names : list | None
            An optional list of the song names to be included in the summary DataFrame. The order
            must corresond to the order of song_ids. The default is None. 
        adjust_tempo : tuple[int, int] | None
            Spotify sometimes misidentifies song tempos as being 2x or 0.5x what they really are.
            Given a tuple of (slowest_tempo, fastest_tempo), check to ensure that all songs tempos
            analyzed fall within that range, and if they do not, double them or halve them to 
            fit the range. This relies on the assumption that all of the songs truly are danceable,
            and that their tempos fall within the given range. The default is (60, 140).
        progress_bar : bool, optional
            Show a progress bar or not. The default is True.

        Returns
        -------
        summary : pd.DataFrame
            A DataFrame containing a subset of the Spotify song analysis information. 

        """
        
        # Define the iterator
        iterator = tqdm(song_ids, desc="Analyzing Songs") if progress_bar else song_ids
        
        # Initialize a dictionary to hold the analyses for a given song
        analyses = {song_id: self._sp.audio_analysis(song_id) 
                    for song_id in iterator}
            
            
        # This should be an input to the function
        desired_info = [
            "tempo",
            "tempo_confidence",
            "key",
            "key_confidence",
            "time_signature", 
            "time_signature_confidence"
        ]    

        # Get a summary of the information for all of these songs in a single dataframe
        summary = pd.concat(
            {info_type: 
                 pd.Series({k:analysis['track'][info_type] for k, analysis in analyses.items()}) 
                 for info_type in desired_info},axis=1
        ).rename_axis("URL")
            
        if song_names is not None:
            summary["name"] = song_names
            
            # Move the name column to the front
            last_column = summary.pop(summary.columns[-1])
            summary.insert(0, last_column.name, last_column)
            
            
        if adjust_tempo is not None:
            # Unpack the bounds
            min_tempo, max_tempo = adjust_tempo
            
            # Add a flag for if the tempo was adjusted
            summary['tempo_adjustment_factor'] = 1.0
            summary.loc[summary['tempo'] < min_tempo, 'tempo_adjustment_factor'] = 2
            summary.loc[summary['tempo'] > max_tempo, 'tempo_adjustment_factor'] = 0.5
           
            # Fit the tempos to the correct range
            summary.loc[summary['tempo'] < min_tempo, "tempo"] *= 2
            summary.loc[summary['tempo'] > max_tempo, "tempo"] /= 2
            
        return summary
        
# -------------------------------------------------------------------------------------------------
# Match Profile 
# -------------------------------------------------------------------------------------------------   
    
    def match_tempo_profile(
            self, 
            tempo_profile: np.ndarray, 
            song_summary_df: pd.DataFrame,
            method: str = "naive",
            plot_results: bool = True,
            ) -> pd.DataFrame:
        """
        Select songs from a given DataFrame such that the tempo of the songs 
        matches a given tempo profile.

        Parameters
        ----------
        tempo_profile : np.ndarray
            A 1D array of target tempo values for each song.
        song_summary_df : pd.DataFrame
            A DataFrame with a "tempo" column which contains the BPM information for the song.
        method : str
            Determine what method to use to approximate the playlist. 
            
            "naive":
                Deterministically select songs by moving 1 target BPM at a time and searching
                through the entire song_summary_df for the song that minimizes the error between 
                the target BPM and the song's BPM. Thus, if there are a limited number of songs in 
                the song_summary_df, the beginning of the playlist is more likely to accurately 
                represent the target profile than the end of the playlist.
            "euclidean":
                Minimize the vertical euclidean distances between the target profile and the 
                achieved profile. Utilizes linear sum assignment to solve the optimization problem.
            "upsampled_euclidean":
                Minimize the vertical euclidean distances between a 50x upsampled version of the 
                target profile and the achieved profile. Utilizes linear sum assignment to solve 
                the optimization problem. Useful when the number of songs in the profile is close 
                to the number of songs in the song_summary_df to get a qualitative representation 
                of what the target profile generally looks like. (ie, when rearranging an existing
                playlist).

        Raises
        ------
        ValueError
            If the tempo profile is longer than the number of songs in the song_summary_df, raise
            an error. 

        Returns
        -------
        selected_songs : pd.DataFrame
            The subset of songs from the original DataFrame ordered to match the profile. 

        """
        
        # plot_results : bool
        #     Create a Matplotlib figure of the target profile and the achieved profile. If the 
        #     target profile is the same length as the song_summary_df, also plots the original
        #     song profile. 
                
        # results : dict[pd.DataFrame, plt.Figure]
        #     A dictionary containing the selected_songs DataFrame and the Figure object if 
        #     plot_results is set to True. 

        
        # Validate the inputs
        if len(tempo_profile) > len(song_summary_df):
            raise ValueError("There are not enough songs to match this profile. Add more songs.")
        
        if method == "naive":
            
            # Initialize a list
            selected_songs = []
            
            # Loop through each position of the tempo profile and find the closest song
            for bpm in tempo_profile:
                min_idx = np.argmin(abs(song_summary_df['tempo'] - bpm))
                
                # Save the closest song
                selected_songs.append(song_summary_df.iloc[min_idx, :])
                
                # Drop the closest song so that it isn't re-selected
                song_summary_df = song_summary_df.drop(index=song_summary_df.index[min_idx])
                
            # Turn the list into a DataFrame
            selected_songs = pd.concat(selected_songs, axis=1).T
            
        elif method == "euclidean" or method == "upsampled_euclidean":
            # https://stackoverflow.com/questions/39016821/minimize-total-distance-between-two-sets-of-points-in-python
            
            if method == "upsampled_euclidean":
                # upsample the target profile so that we get a more faithful representation 
                # of it
                

                
                n_songs = len(tempo_profile)
                
                scale_factor = 50
                # if you don't cut off the last 'scale_factor' points, it looks weird. idk why
                tempo_profile = np.interp(
                    x=np.linspace(0, n_songs, n_songs*scale_factor),
                    xp=np.arange(n_songs),
                    fp=tempo_profile,
                )[:-scale_factor]
                

            # Calculate the 'cost' (euclidean distance which equals absolute error) of each
            # combination of points
            
            cost = cdist(
                tempo_profile.reshape(-1,1),
                song_summary_df['tempo'].to_numpy().reshape(-1,1), 
                metric="euclidean",
            )
            
            # Optimize the song assignment
            _, song_idxs = linear_sum_assignment(cost)
            
            # Set the song assignment
            selected_songs = song_summary_df.iloc[song_idxs, :]
            
        return selected_songs

# -------------------------------------------------------------------------------------------------
# Save Playlist
# -------------------------------------------------------------------------------------------------   

    def save_playlist(
            self, 
            playlist_name: str,
            playlist_songs: list[str],
            description: str = "Created by the DJAssistant.",
            cover_image_b64: bytes | None = None,
            verbose: bool = True,
            ):
        
        # Validate the inputs
        if len(playlist_name) > 100:
            print("Truncating playlist name to 100 characters.")
            playlist_name = playlist_name[:100]
        
        # Get your user_id
        user_id = self._sp.me()['id']
        
        # Create the playlist and get it's URL
        playlist_id = self._sp.user_playlist_create(
            user=user_id, 
            name=playlist_name, 
            description=description
        )['external_urls']['spotify']
        
        # Add the songs to the playlist
        self._sp.playlist_add_items(playlist_id, playlist_songs)
        
        if cover_image_b64 is not None:
            self._sp.playlist_upload_cover_image(playlist_id, cover_image_b64)
        
        if verbose:
            print(f"Successfully created playlist: {playlist_name}")
        
        

    