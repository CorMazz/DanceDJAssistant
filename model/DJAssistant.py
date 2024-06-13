# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:28:03 2023

@author: mazzac3

TODO:
    
    Add the capability to create interactive plotly plots describing the playlist. Hovertext containing key info about
    the songs.
"""

import os
import io
import base64
import spotipy
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from spotipy.oauth2 import SpotifyOAuth
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from tenacity import retry, stop_after_attempt, wait_exponential


class DanceDJ:
    
    def __init__(
            self, 
            scope="playlist-modify-public playlist-modify-private ugc-image-upload", 
            retry_config: dict | None = None,
            # db_url: str | os.PathLike | None = None,
        ):
        """
        Instantiates a connection to the Spotify API. See the Spotipy documentation to learn 
        about the authorization scope.
        
        retry_config : dict, optional
            A dict of kwargs fed to the Tenacity retry decorator wrapping Spotipy's finnicky audio_analysis method. 
            Default is None, which leads to the following retry settings: 
                {
                    'stop': stop_after_attempt(3),
                    'wait': wait_exponential(multiplier=1, min=4, max=10)
                }
        """
        # db_url : str | False | None, optional
        #     A url for the SQL database to connect to. The database will be used as a persistent cache of song 
        #     information to limit calls to the Spotipy API. 
        #         If a string | os.PathLike:
        #             Connects to the existing database at this URL and adds a table. Once the analyze songs method is 
        #             called. If there is no existing database, creates a new one.
        #         If None: 
        #             Does not connect to a database.

        
        self._sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))
        
        if not isinstance(retry_config, (dict, type(None))):
            raise TypeError(f"retry_config is of type {type(retry_config)} when it should be a dict or None.")
        self.retry_config = retry_config or {
            'stop': stop_after_attempt(3),
            'wait': wait_exponential(multiplier=1, min=4, max=10)
        }
        
        # if not isinstance(db_url, (str, os.PathLike, type(None))):
        #     raise TypeError(f"db_url is of type {type(db_url)} when it should be a str, os.PathLike, or None.")
        # elif isinstance(db_url, str):
        #     db_url = Path(db_url)
            
        # if db_url is not None:
            
        
########################################################################################################################
# Primary Methods
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# Parse Playlist
# ----------------------------------------------------------------------------------------------------------------------

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
        adjust_tempo: tuple[int, int] | None = (60, 130),
        desired_info: tuple[str] = (
            "tempo",
            "tempo_confidence",
            "key",
            "key_confidence",
            "time_signature", 
            "time_signature_confidence"
        ),
        progress_bar: bool = True,
        ) -> pd.DataFrame | str:
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
            and that their tempos fall within the given range. The default is (60, 130).
        desired_info: tuple[str], optional
            The names of the columns to keep in the returned DataFrame. Columns to select from are provided by the 
            Spotipy API. https://developer.spotify.com/documentation/web-api/reference/get-audio-analysis. 
            The default is
            (
                "tempo",
                "tempo_confidence",
                "key",
                "key_confidence",
                "time_signature", 
                "time_signature_confidence"
            )
        
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
            summary = self.adjust_tempo(summary, adjust_tempo)
            
        return summary
    
    # TODO: Add a function which analyzes the playlist and gives a summary of it's stats, such as total length, bpm 
    # quartiles, a bpm histogram, 
    
# ----------------------------------------------------------------------------------------------------------------------
# Match Profile 
# ----------------------------------------------------------------------------------------------------------------------
    
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
                # if you don't cut off the last 'scale_factor' # of points, it looks weird. idk why
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
            cover_image_b64: bytes | plt.Figure | None = None,
            verbose: bool = True,
            ):
        """
        Save the playlist by uploading it to your Spotify account. Can optionally add a playlist
        description and cover image.

        Parameters
        ----------
        playlist_name : str
            The name of the playlist. There is a 100 character limit.
        playlist_songs : list[str]
            A list of strings of unique Spotify song identifiers (URLs, URIs, IDs). Not names.
        description : str, optional
            A description for the playlist. The default is "Created by the DJAssistant.".
        cover_image_b64 : bytes | plt.Figure | None, optional
            A b64 byte string containing the image, or a Matplotlib figure that will be converted
            to a b64 byte string. Ideally, make the figure square. 5x5 seems to work well. The max
            size is 256 kB. The default is None.
        verbose : bool, optional
            Print status to console. The default is True.

        Returns
        -------
        None.

        """
        
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
        
        # Upload the cover image
        if cover_image_b64 is not None:
            
            # Check if it is a matplotlib figure that needs to be converted
            if isinstance(cover_image_b64, plt.Figure):
                with io.BytesIO() as buffer:
                    cover_image_b64.savefig(buffer, format='jpeg')
                    
                    # Check if the buffer size is within the limit (256 KB)
                    if buffer.tell() >= 256 * 1024:
                        raise OverflowError("The given figure is larger than 256 kB.")
               
                    buffer.seek(0)  # Reset buffer position to start
                    cover_image_b64 = base64.b64encode(buffer.getvalue()) # Encode the figure
            
            # upload the figure
            self._sp.playlist_upload_cover_image(playlist_id, cover_image_b64)
        
        if verbose:
            print(f"Successfully created playlist: {playlist_name}")
        
# ----------------------------------------------------------------------------------------------------------------------
# Adjust Tempo
# ----------------------------------------------------------------------------------------------------------------------
        

    def adjust_tempo(self, playlist: pd.DataFrame, tempo_bounds: tuple[int, int]) -> pd.DataFrame:
        
        # Unpack the bounds
        min_tempo, max_tempo = tempo_bounds
        
        # Add a flag for if the tempo was adjusted
        playlist['tempo_adjustment_factor'] = 1.0
        playlist.loc[playlist['tempo'] < min_tempo, 'tempo_adjustment_factor'] = 2
        playlist.loc[playlist['tempo'] > max_tempo, 'tempo_adjustment_factor'] = 0.5
       
        # Fit the tempos to the correct range
        playlist.loc[playlist['tempo'] < min_tempo, "tempo"] *= 2
        playlist.loc[playlist['tempo'] > max_tempo, "tempo"] /= 2
        return playlist
    
# -------------------------------------------------------------------------------------------------
# Robust Audio Analysis
# -------------------------------------------------------------------------------------------------  
    
    def __get_retry_decorator(self):
        """Enables us to have the user feed in retry options via kwargs to the DanceDJ constructor."""
        return retry(**self.retry_config)

    def robust_audio_analysis(self, song_id):
        """Wraps Spotipy's audio_analysis method with a Tenacity decorator that makes it retry polling the API if it 
        times out. The retry parameters can be set in the DanceDJ constructor."""
        @self.__get_retry_decorator()
        def _fetch():
            return self._sp.audio_analysis(song_id)
        return _fetch()
    
# ----------------------------------------------------------------------------------------------------------------------
# Generate Sinusoidal Profile
# ----------------------------------------------------------------------------------------------------------------------

    def generate_sinusoidal_profile(
            self, 
            tempo_bounds: tuple[int, int],
            n_cycles: float, 
            horizontal_shift: float, 
            n_songs: int
        ) -> np.ndarray:
        """
        Define a sinusoidal tempo profile for a given number of songs.
    
        This function generates a sinusoidal profile of target tempos for a given 
        number of songs. The sinusoidal function is defined by the amplitude and 
        mean of the tempo bounds, and it is modulated over a specified number of 
        cycles with an optional horizontal shift.
    
        Parameters
        ----------
        tempo_bounds : tuple of int
            A tuple containing the minimum and maximum tempo values.
        n_cycles : float
            The number of cycles over which the sinusoidal profile should repeat.
        horizontal_shift : float
            The horizontal shift applied to the sinusoidal function.
        n_songs : int
            The number of songs in the generated profile.
    
        Returns
        -------
        np.ndarray
            An array containing the sinusoidal tempo profile.
        """
        # Define a target tempo profile for n songs
        amplitude = tempo_bounds[1] - tempo_bounds[0]
        mean = np.mean(tempo_bounds)
        song_idx = np.linspace(0, 1, n_songs)
        return amplitude*np.sin( n_cycles * 2 * np.pi * (song_idx + horizontal_shift*(n_songs/n_cycles))) + mean
    
# ----------------------------------------------------------------------------------------------------------------------
# Plot Profile
# ----------------------------------------------------------------------------------------------------------------------

    def plot_profile(
            self, 
            profile: np.ndarray,
            fig_kwargs: dict | None = None,
            ax_plot_kwargs: dict | None = None,
        ) -> (plt.Figure, plt.Axes):
        """
        Plot the target tempo profile.
    
        This function creates a plot of the given target tempo profile using 
        matplotlib. The plot can be customized with additional keyword arguments 
        for the figure and axes.
    
        Parameters
        ----------
        profile : np.ndarray
            An array containing the tempo profile to be plotted.
        fig_kwargs : dict or None, optional
            A dictionary of keyword arguments to be passed to `plt.subplots` 
            for figure customization. Default is None.
        ax_plot_kwargs : dict or None, optional
            A dictionary of keyword arguments to be passed to `ax.plot` 
            for axis customization. Default is None.
    
        Returns
        -------
        fig : plt.Figure
            The matplotlib figure object containing the plot.
        ax : plt.Axes
            The matplotlib axes object containing the plot.
    
        Examples
        --------
        >>> dj = DanceDJ()
        >>> profile = dj.define_sinusoidal_profile((60, 120), 3, 0.5)
        >>> fig, ax = dj.plot_profile(profile)
        >>> plt.show()
    
        Notes
        -----
        The function uses default plot settings, which can be overridden by 
        providing `fig_kwargs` and `ax_plot_kwargs`. If no customization is 
        provided, the plot will use the default settings:
        
        - `fig_kwargs`: empty dictionary
        - `ax_plot_kwargs`: {'label': 'Target Profile', 'color': 'k', 'marker': 'o', 'linestyle': '--'}
    
        The function also validates the types of the input variables before 
        proceeding with plotting.
        """
        
        self.__validate_variable_types([
            ("profile", profile, np.ndarray),
            ("fig_kwargs", fig_kwargs, (dict, type(None))),
            ("ax_plot_kwargs", ax_plot_kwargs, (dict, type(None)))
        ])
        
        fig_kwargs = fig_kwargs or {}
        
        # Define default ax_plot_kwargs and update with function inputs
        ax_plot_kwargs = dict(label="Target Profile", color="k", marker="o", linestyle="--") | (ax_plot_kwargs or {})
        
        # Plot the target profile
        fig, ax = plt.subplots(**fig_kwargs)
        ax.plot(profile, **ax_plot_kwargs)
        ax.set_ylabel("Tempo (BPM)")
        ax.set_xlabel("Song Number")
        ax.set_title("Target Song Profile")
        ax.legend()
        
        return fig, ax
        
# ----------------------------------------------------------------------------------------------------------------------
# Validate Variable Types
# ----------------------------------------------------------------------------------------------------------------------
        
    @staticmethod
    def __validate_variable_types(variables):
        """
        Validates the types of variables against their allowed types.
    
        Parameters
        ----------
        variables : list of tuples
            A list where each tuple contains:
            - variable_name : str
                The name of the variable to be checked.
            - variable : any
                The variable to be checked.
            - allowed_types : tuple of types
                The types that the variable is allowed to be.
            - error_message : str, optional
                A custom error message to be used if the variable is not of the allowed types.
    
        Raises
        ------
        TypeError
            If a variable is not of the allowed types.
    
        Examples
        --------
        #>>> import numpy as np
        #>>> data = np.array([1, 2, 3])
        #>>> validate_variable_types([('data', data, (np.ndarray, type(None)))])
    
        #>>> validate_variable_types([('data', data, (list, type(None)))])
        Traceback (most recent call last):
            ...
        TypeError: data is of type <class 'numpy.ndarray'> when it should be one of (<class 'list'>, <class 'NoneType'>).
    
        #>>> validate_variable_types([('data', data, (list, type(None)), "CUSTOM ERROR MESSAGE HERE.")])
        Traceback (most recent call last):
            ...
        TypeError: data is of type <class 'numpy.ndarray'> CUSTOM ERROR MESSAGE HERE.
        """
        for entry in variables:
            variable_name = entry[0]
            var = entry[1]
            allowed_types = entry[2]
            error_message = entry[3] if len(entry) > 3 else None
    
            if not isinstance(var, allowed_types):
                if error_message:
                    raise TypeError(f"{variable_name} is of type {type(var)} {error_message}")
                else:
                    raise TypeError(
                        f"{variable_name} is of type {type(var)} when it should be one of {allowed_types}."
                    )