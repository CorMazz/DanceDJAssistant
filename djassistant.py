# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:28:03 2023

@author: mazzac3

Possible Improvements:
    
    Add the capability to create interactive plotly plots describing the playlist. Hovertext containing key info about
    the songs.

    Refactor this to have methods which return Dataclasses where necesssary. 
            
"""

import os
import io
import base64
import spotipy
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import plotly.express as px
import matplotlib.pyplot as plt
from dataclasses import dataclass
import plotly.graph_objects as go
from spotipy.oauth2 import SpotifyOAuth
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from tenacity import retry, stop_after_attempt, wait_exponential



try:
    from sqlalchemy import create_engine, Column, Integer, String, inspect, Float
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    from contextlib import contextmanager
    db_enabled=True
except ImportError:
    db_enabled=False

NoneType = type(None)

########################################################################################################################
# Optional Database Functionality
########################################################################################################################

if db_enabled:
    
    # Define the Base class for declarative models
    Base = declarative_base()
    
    # Define an example model (table)
    class Song(Base):
        """Store songs in the database. Each song can be identiied by its URL/I"""
        __tablename__ = 'djassistant_songs'
        url = Column(String(300), primary_key=True)
        tempo = Column(Float)
        duration = Column(Float)
        key = Column(Integer)
        time_signature = Column(Integer)
        
        def to_dict(self, include_url: bool = True):
            """
            

            Parameters
            ----------
            include_url : bool, optional
                Include the URL as a key-value pair in the dict. The default is True.

            Returns
            -------
            return_val : TYPE
                A dictionary of the parameters in the class. 

            """
            return_val = dict(tempo=self.tempo,duration=self.duration, key=self.key, time_signature=self.time_signature)
            
            if include_url:
                return_val.update(dict(url=self.url))
            return return_val


########################################################################################################################
# Optional Database Functionality
########################################################################################################################

class DanceDJ:
    
    def __init__(
            self, 
            scope="playlist-modify-public playlist-modify-private ugc-image-upload", 
            spotify_obj: spotipy.Spotify | None = None,
            db_session: None = None,
            retry_config: dict | None = None,
        ):
        """
        Initializes the class instance.

        spotify_obj : spotipy.Spotify | None = None
            A spotipy.Spotify object with the following scope:
                "playlist-modify-public playlist-modify-private ugc-image-upload"
                
        db_session : sqlalchemy.Session | None, optional
            An optional sqlalchemy.Session object to use as a database to implement song caching. Creates a table called
            djassistant_songs if it does not already exist. The default is None, meaning no caching is implemented. 
            Currently the database stores a song URL as the primary key, it's tempo in BPM, and it's duration.      
            
        retry_config : dict, optional
            A dict of kwargs fed to the Tenacity retry decorator wrapping Spotipy's finnicky audio_analysis method. 
            Default is None, which leads to the following retry settings: 
                {
                    'stop': tenacity.stop_after_attempt(3),
                    'wait': tenacity.wait_exponential(multiplier=1, min=4, max=10)
                }
                

        """

        self.__validate_variable_types([
            ("scope", scope, str),
            ("retry_config", retry_config, (dict, NoneType)),
            ("spotify_obj", spotify_obj, (spotipy.Spotify, NoneType)),
        ])

        self._sp = spotify_obj
        
        if not isinstance(retry_config, (dict, NoneType)):
            raise TypeError(f"retry_config is of type {type(retry_config)} when it should be a dict or None.")
        self._retry_config = retry_config or {
            'stop': stop_after_attempt(3),
            'wait': wait_exponential(multiplier=1, min=4, max=10)
        }
        
        self._db_session = db_session
        if db_enabled and db_session is not None:
            # Create the table if it doesn't already exist in the db
            engine = self._db_session.get_bind() 
            inspector = inspect(engine)
            if not inspector.has_table("djassistant_songs"):
                Base.metadata.create_all(engine)
                
        # Key mapping to go from spotify integer keys to traditional letter keys
        self.key_mapping = {
            0: 'C',
            1: 'C♯/D♭',
            2: 'D',
            3: 'D♯/E♭',
            4: 'E',
            5: 'F',
            6: 'F♯/G♭',
            7: 'G',
            8: 'G♯/A♭',
            9: 'A',
            10: 'A♯/B♭',
            11: 'B'
        }
                    
            
        
########################################################################################################################
# Primary Methods
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# Parse RawPlaylist
# ----------------------------------------------------------------------------------------------------------------------

    def parse_playlist(
            self,
            playlist_id: str, 
            max_tracks: int = 5e3,
            return_api_response: bool = False
        ) -> dict | tuple[dict, dict]:
        """
        Parse a given playlist and grab the IDs and names of it's songs.
    
        Parameters
        ----------
        playlist_id : str
            The Spotify ID of a playlist. Can be a url. 
        max_tracks : int, optional
            The maximum number of tracks to parse from the playlist (default is 5000).
        return_api_response : bool, optional
            Return the response of the call to Spotify's Get Playlist API. Default is False
        Returns
        -------
        parsed_songs : dict
            A dictionary where the key is the url for a song, and the value is the song name. 
            This was on purpose, since urls are unique, as keys need to be. 
    
        if return_api_response:
            Returns a tuple where the second object is a dictionary with Spotify's API response. Some useful parameters 
            in that dictionary may be name and description. 
        """
        
        if self._sp is None:
            raise SpotifyNotActivatedError("This DanceDJ instance was initialized without a Spotify object.")
        
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
            
        if return_api_response:
            return parsed_songs, playlist
            
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
            "key",
            "time_signature", 
            "duration",
            
        ),
        progress_bar: bool = True,
        load_from_db: bool = True,
        save_to_db: bool = True,
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
                "key",
                "time_signature", 
                "duration",
            )
        
        progress_bar : bool, optional
            Show a progress bar or not. The default is True.
        load_from_db : bool, optional
            Loads songs that are already in the database instead of re-analyzing them, if a session was provided when 
            constructing the DanceDJ class. Default is True.
        save_to_db : bool, optional
            Save new songs to the database, if a session was provided when constructing the DanceDJ class. Default 
            is True.


        Returns
        -------
        summary : pd.DataFrame
            A DataFrame containing a subset of the Spotify song analysis information. 

        """
        
        # If the session exists, search for the songs that are already within a table labeled "songs" and grab the desired
        # info from that table. If the desired info isn't in the table, or a song isn't in the table, then perform the 
        # robust analysis below.
        
        if self._db_session is not None:
            # TODO: replace the outdated query API with the prefered API
            cached_songs = self._db_session.query(Song).filter(Song.url.in_(song_ids)).all() if load_from_db else []
            
            # Extract useful parameters as dict
            cached_songs =  {song.url: song.to_dict(include_url=False)
                for song in cached_songs
            }
            
            cached_song_summary=pd.DataFrame(cached_songs).T
            
            new_songs = set(song_ids) - set(cached_songs.keys())
        else:
            cached_song_summary = pd.DataFrame()
            new_songs = song_ids
        
        # Make the tqdm iterator count the total songs starting from the number found in the cache
        iterator = (
            tqdm(new_songs, desc="Analyzing Songs", initial=len(cached_song_summary), total=len(song_ids)) 
            if progress_bar else new_songs
        )
        
        # Initialize a dictionary to hold the analyses for a given song
        
        if new_songs and self._sp is None:
            raise SpotifyNotActivatedError("This DanceDJ instance was initialized without a Spotify object.")
        
        analyses = {}
        
        for song_id in iterator:
            try:
                analyses[song_id] = self.robust_audio_analysis(song_id) 
            
                # Store the new songs in the database
                if self._db_session is not None and save_to_db:
                    
                    # Prepare data for database storage
                    
                    song = Song(**{
                        'url': song_id, 
                        'tempo': analyses[song_id]['track']['tempo'],
                        'duration': analyses[song_id]['track']['duration'],
                        'key': analyses[song_id]['track']['key'],
                        'time_signature': analyses[song_id]['track']['time_signature'],
                        }
                    )
                    
                    self._db_session.add(song)
            except Exception as e:
                if self._db_session is not None and save_to_db:
                    self._db_session.commit()
                raise e
            
        if self._db_session is not None and save_to_db:
            self._db_session.commit()
            
        # Get a summary of the information for all of these songs in a single dataframe
        new_song_summary = pd.concat(
            {info_type: 
                 pd.Series({k:analysis['track'][info_type] for k, analysis in analyses.items()}) 
                 for info_type in desired_info},axis=1
        ).rename_axis("URL")
            
        all_songs = pd.concat([new_song_summary, cached_song_summary])
            
        # Reorder the songs back to their original order
        all_songs = all_songs.loc[song_ids]
            
        if song_names is not None:
            all_songs["name"] = song_names
            
            # Move the name column to the front
            last_column = all_songs.pop(all_songs.columns[-1])
            all_songs.insert(0, last_column.name, last_column)
            
            
        if adjust_tempo is not None:
            all_songs = self.adjust_tempo(all_songs, adjust_tempo)
            
        return all_songs
    
    # TODO: Add a function which analyzes the playlist and gives a summary of it's stats, such as total length, bpm 
    # quartiles, a bpm histogram, 
    
# ----------------------------------------------------------------------------------------------------------------------
# Match Profile 
# ----------------------------------------------------------------------------------------------------------------------
    
    def match_tempo_profile(
            self, 
            tempo_profile: np.ndarray, 
            song_summary_df: pd.DataFrame,
            method: str = "upsampled_euclidean",
            plot_results: bool = True,
            ) -> pd.DataFrame:
        """
        Select songs from a given DataFrame such that the tempo of the songs 
        matches a given tempo profile.

        Parameters
        ----------
        tempo_profile : np.ndarray
            A 1D array of target tempo values for each song. This can be longer than the number of songs, since the 
            Euclidean and upsampled Euclidean methods will try to rearrange the playlist such that the global error is
            minimized. 
        song_summary_df : pd.DataFrame
            A DataFrame with a "tempo" column which contains the BPM information for the song.
        method : str
            Determine what method to use to approximate the playlist. Default is naive.
            
            "naive":
                Deterministically select songs by moving 1 target BPM at a time and searching
                through the entire song_summary_df for the song that minimizes the error between 
                the target BPM and the song's BPM. Thus, if there are a limited number of songs in 
                the song_summary_df, the beginning of the playlist is more likely to accurately 
                represent the target profile than the end of the playlist.
            "global_naive":
                Attempts to match the playlist to the profile by sorting the profile and the playlist in ascending order
                and then assigning a 1 to 1 correspondence from song to playlist location. Doesn't really work to well
                but I left it for posterity. 
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


        Returns
        -------
        selected_songs : pd.DataFrame
            The subset of songs from the original DataFrame ordered to match the profile. 

        """
        
        if tempo_profile.ndim > 1:
            raise ValueError("tempo_profile must be a 0D or 1D Numpy array but it is of ndim = {tempo_profile.ndim}")
        
        
        if method == "naive":
            
            if len(tempo_profile) != len(song_summary_df):
                raise ValueError("The number of points in the profile must exactly match the number of songs in the "
                                 "playlist for the naive method.")
            
            # Initialize a list
            selected_songs = []
            
            # Loop through each position of the tempo profile and find the closest song
            for bpm in tempo_profile:
                if not len(song_summary_df):
                    break
                
                min_idx = np.argmin(abs(song_summary_df['tempo'] - bpm))
                
                # Save the closest song
                selected_songs.append(song_summary_df.iloc[min_idx, :])
                
                # Drop the closest song so that it isn't re-selected
                song_summary_df = song_summary_df.drop(index=song_summary_df.index[min_idx])
                
            # Turn the list into a DataFrame
            selected_songs = pd.concat(selected_songs, axis=1).T
        
        elif method == "global_naive":
                        
            if len(tempo_profile) != len(song_summary_df):
                raise ValueError("The number of points in the profile must exactly match the number of songs in the "
                                 "playlist for the global naive method.")
                
            # Sort the sine curve by magnitude, sort the songs by magnitude, match them up to each other
            selected_songs = (
                song_summary_df
                .sort_values(by="tempo", ascending=True)
                .iloc[np.argsort(tempo_profile), :]
            )
            
            
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
            
        else:
            raise ValueError("Check the documentation for appropriate values for the 'method' parameter.")
            
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
            raise_errors: bool = True,
            verbose: bool = True,
            ) -> str:
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
        raise_errors : bool, optional
            Will raise an OverflowError if the image provided is larger than 256 kB. The default is True.
        verbose : bool, optional
            Print status to console. The default is True.

        Returns
        -------
        The playlist URL

        """
        
        if self._sp is None:
            raise SpotifyNotActivatedError("This DanceDJ instance was initialized without a Spotify object.")
        
        
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
            
            attempt_image_upload = True
            # Check if it is a matplotlib figure that needs to be converted
            if isinstance(cover_image_b64, plt.Figure):
                with io.BytesIO() as buffer:
                    cover_image_b64.savefig(buffer, format='jpeg')
                    
                    # Check if the buffer size is within the limit (256 KB)
                    if buffer.tell() <= 256 * 1024:
                        buffer.seek(0)  # Reset buffer position to start
                        cover_image_b64 = base64.b64encode(buffer.getvalue()) # Encode the figure
                    elif raise_errors:
                        raise OverflowError("The given figure is larger than 256 kB.")
                    else:
                        attempt_image_upload = False
                        
                        if verbose:
                            print("Cover image upload not attempted because the image is larger than 256 kB.")
                
            if attempt_image_upload:
                self._sp.playlist_upload_cover_image(playlist_id, cover_image_b64)
        
        if verbose:
            print(f"Successfully created playlist: {playlist_name}")
        
        return playlist_id
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
        return retry(**self._retry_config)

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
            n_songs: int,
            damping_coefficient: float = 0,
            n_points: int | None = None
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
        damping_coefficient: float, optional
            The c in the equation y = Aexp(-cx) to be multiplied by the sinusoidal profile to implement damping.
            This is useful for making a playlist that becomes one medium tempo as the night goes on. By default is 0,
            which is no damping. 
        n_points : int or None, optional
            The number of points to put in the sinusoidal profile. Default is None, which is equivalent to n_songs. 
            Must be greater than n_songs if provided, or will default to n_songs. 
    
        Returns
        -------
        np.ndarray
            An array containing the x-values and y-values of the sinusoidal tempo profile. If n_points is None, the
            x-values (first column) are just the indices of the songs. If n_points is greater than n_songs then the 
            x-values are evenly spread between 1 and the number of songs. 
        """
        
        if n_points is None or n_points < n_songs:
            n_points = n_songs
        
        # Define a target tempo profile for n songs
        amplitude = (tempo_bounds[1] - tempo_bounds[0]) / 2
        mean = np.mean(tempo_bounds)
        x = np.linspace(0, n_songs - 1, n_points)
        y = amplitude * np.sin(2 * np.pi * n_cycles * (x / (n_songs - 1)) + horizontal_shift * 2 * np.pi)
        y *= np.exp( - damping_coefficient*x)
        y += mean
        
        return np.column_stack((x + 1, y))
        
# ----------------------------------------------------------------------------------------------------------------------
# Plot Profile Matplotlib
# ----------------------------------------------------------------------------------------------------------------------

    def plot_profile_matplotlib(
            self, 
            *,
            target_profile: np.ndarray | None = None,
            analyzed_playlist: pd.DataFrame | None = None,
            fig_kwargs: dict | None = None,
            target_profile_kwargs: dict | None = None,
            analyzed_playlist_kwargs: dict | None = None,
            fig: plt.Figure | None = None,
            ax: plt.Axes | None = None,
        ) -> (plt.Figure | None, plt.Axes):
        """
        Plot the target tempo profile or the current playlist profile.
    
        This function creates a plot of the given target tempo profile using 
        matplotlib. The plot can be customized with additional keyword arguments 
        for the figure and axes.
    
        Parameters
        ----------
        target_profile : np.ndarray or None, optional
            An 1 or 2D array containing the tempo profile to be plotted. If 2D, first column is x, second is y. If 1D,
            assumes that the x-values are defined by np.arange(1, len(target_profile))
        analyzed_playlist : pd.DataFrame or None, optional
            A DataFrame with information about the analyzed playlist. At minimum needs to have a "tempo" column. 
            If "tempo_adjustment_factor" is in the DataFrame, will plot those points as red to indicate uncertainty in
            their tempo. 
        fig_kwargs : dict or None, optional
            A dictionary of keyword arguments to be passed to `plt.subplots` 
            for figure customization. Default is None.
        target_profile_kwargs : dict or None, optional
            A dictionary of keyword arguments to be passed to `ax.plot` 
        analyzed_playlist_kwargs : dict or None, optional
            A dictionary of keyword arguments to be passed to `ax.plot` 
            for axis customization. Default is None.
        fig : plt.Figure or None, optional
            A Figure object to create the plot on. Technically does not need the figure, it is simply returned. Axes are
            required, but if no figure is provided, the return signature becomes (None, ax).
        ax : plt.Axes or None, optional
            Axes object to plot on. If None is passed, creates the figure and the axes.
    
        Returns
        -------
        fig : plt.Figure | None
            The matplotlib figure object containing the plot, or None if axes was fed but no figure was fed. 
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
        providing `fig_kwargs` and `target_profile_kwargs`. If no customization is 
        provided, the plot will use the default settings:
        
        - `fig_kwargs`: empty dictionary
        - `target_profile_kwargs`: {'label': 'Target Profile', 'color': 'k', 'marker': 'o', 'linestyle': '--'}
    
        The function also validates the types of the input variables before proceeding with plotting.
        """
        
        self.__validate_variable_types([
            ("target_profile", target_profile, (np.ndarray, NoneType)),
            ("analyzed_playlist", analyzed_playlist, (pd.DataFrame, NoneType)),
            ("fig_kwargs", fig_kwargs, (dict, NoneType)),
            ("target_profile_kwargs", target_profile_kwargs, (dict, NoneType)),
            ("analyzed_playlist_kwargs", analyzed_playlist_kwargs, (dict, NoneType)),
            ("fig", fig, (plt.Figure, NoneType)),
            ("ax", ax, (plt.Axes, NoneType))
        ])
        
        fig_kwargs = fig_kwargs or {}
        
        if ax is None:
            fig, ax = plt.subplots(**fig_kwargs)
        
        if target_profile is not None:
            # Define default target_profile_kwargs and update with function inputs
            target_profile_kwargs = (
                dict(label="Target Profile", color="k", marker="o", linestyle="--") | (target_profile_kwargs or {})
            )
            if target_profile.ndim == 1:
                ax.plot(target_profile, **target_profile_kwargs)
            elif target_profile.ndim == 2:
                ax.plot(target_profile[:, 0], target_profile[:, 1], **target_profile_kwargs)
            else:
                raise ValueError("Target profile must not be greater than 2D. ")
        
        if analyzed_playlist is not None:
            if analyzed_playlist.get("tempo") is None:
                raise KeyError(
                    "There is no 'tempo' column in the analyzed playlist DataFrame. This is required if that DataFrame "
                    "is provided."
                )
                
            analyzed_playlist_kwargs = (
                dict(label="RawPlaylist Profile", color="b", marker="^", linestyle="-") | (analyzed_playlist_kwargs or {})
            )
            
            ax.plot(
                np.arange(1, len(analyzed_playlist) + 1),
                analyzed_playlist.get("tempo").to_numpy(),
                **analyzed_playlist_kwargs
            )

            # Highlight adjusted tempos which may be incorrect.
            if "tempo_adjustment_factor" in analyzed_playlist:
                # Points where tempo_adjustment_factor is not 1
                adjustment_mask = analyzed_playlist["tempo_adjustment_factor"] != 1
                ax.plot(
                    np.arange(1, len(analyzed_playlist) + 1)[adjustment_mask], 
                    analyzed_playlist["tempo"][adjustment_mask], 
                    **(
                        analyzed_playlist_kwargs 
                        | dict(color="r", label="Adjusted Tempo Songs", linestyle="", markersize=8)
                    ), 
                )
            
        ax.set_ylabel("Tempo (BPM)")
        ax.set_xlabel("Song Number")
        ax.legend(loc="upper right")
        
        return fig, ax
        
      
# ----------------------------------------------------------------------------------------------------------------------
# Plot Profile Plotly
# ----------------------------------------------------------------------------------------------------------------------  
    
    def plot_playlist_plotly(
            self, 
            analyzed_playlist: pd.DataFrame | None = None, 
            target_profile: np.ndarray | None = None,
            target_profile_kwargs: dict | None = None,
            update_layout_kwargs: dict | None = None,
            fig: go.Figure | None = None,
        ) -> go.Figure:
        """
        Plot the analyzed playlist tempo profile and optionally a target profile using Plotly.
    
        This function creates a Plotly plot of the given analyzed playlist's tempo profile. Optionally, a target tempo
        profile can be added to the plot for comparison. The plot can be customized with additional keyword arguments 
        for the target profile and layout.
    
        Parameters
        ----------
        analyzed_playlist : pd.DataFrame or None, optional
            A DataFrame with information about the analyzed playlist. At a minimum, it needs to have columns for 
            'duration', 'tempo', 'key', and 'tempo_adjustment_factor'. Other columns will be used for hover text.
        target_profile : np.ndarray or None, optional
            An 1 or 2D array containing the tempo profile to be plotted. If 2D, the first column is x, the second is y.
            If 1D, assumes that the x-values are defined by np.arange(1, len(target_profile)).
        target_profile_kwargs : dict or None, optional
            A dictionary of keyword arguments to be passed to `go.Scatter` for customizing the target profile plot.
            Default is None.
        update_layout_kwargs : dict or None, optional
            A dictionary of keyword arguments to be passed to `fig.update_layout` for customizing the plot layout.
            Default is None.
        fig : go.Figure or None, optional
            A Plotly Figure object to create the plot on. If None is provided, a new figure will be created.
    
        Returns
        -------
        fig : go.Figure
            The Plotly figure object containing the plot.
    
        Examples
        --------
        >>> dj = DanceDJ()
        >>> playlist = dj.analyze_playlist('my_playlist')
        >>> fig = dj.plot_playlist_plotly(playlist)
        >>> fig.show()
    
        Notes
        -----
        The function uses default plot settings, which can be overridden by providing `target_profile_kwargs` and 
        `update_layout_kwargs`. If no customization is provided, the plot will use the default settings:
        
        - `target_profile_kwargs`: {'name': 'Target Profile', 'mode': 'lines', 'line': {'color': 'black'}}
        - `update_layout_kwargs`: {
            'xaxis_title': 'Song Index', 
            'yaxis_title': 'Tempo (BPM)',
            'title': 'Playlist Tempo Analysis',
            'template': 'plotly_white'
            }
    
        The function also validates the types of the input variables before proceeding with plotting.
        """
        
        self.__validate_variable_types([
            ("target_profile", target_profile, (np.ndarray, NoneType)),
            ("analyzed_playlist", analyzed_playlist, (pd.DataFrame, NoneType)),
            ("update_layout_kwargs", update_layout_kwargs, (dict, NoneType)),
            ("target_profile_kwargs", target_profile_kwargs, (dict, NoneType)),
            ("fig", fig, (go.Figure, NoneType)),
        ])
        
        # Create teh figure object if it was not provided
        fig = fig or go.Figure()
        
        if analyzed_playlist is not None:
            playlist = analyzed_playlist.copy()
            
            # Calculate the cumulative runtime of the songs
            playlist['Song Start Time'] = playlist['duration'].cumsum()
            
            # Rename columns to have title capitalization and units
            playlist.rename(columns={
                'duration': 'Duration (mm:ss)',
                'tempo': 'Tempo (BPM)',
                'key': 'Key',
                'tempo_adjustment_factor': 'Tempo Adjustment Factor'
            }, inplace=True)
            
            # Format the Song Start Time into a string of hh:mm:ss
            playlist['Song Start Time'] = pd.to_datetime(playlist['Song Start Time'], unit='s').dt.strftime('%H:%M:%S')
            
            # Convert the duration into a string of mm:ss
            playlist['Duration (mm:ss)'] = (pd.to_datetime(playlist['Duration (mm:ss)'], unit='s')
                                            .dt.strftime('%M:%S'))
            
            # Convert the key from an integer into a string using your mapping dictionary
            playlist["Key"] = playlist["Key"].replace(self.key_mapping)
            
            # Prepare data for the plot
            x_values = np.arange(1, len(playlist) + 1)
            y_values = playlist['Tempo (BPM)']
            colors = ['#EF553B' if taf != 1 else '#636EFA' for taf in playlist['Tempo Adjustment Factor']]
            symbols = ['diamond' if taf != 1 else 'circle' for taf in playlist['Tempo Adjustment Factor']]
            
            hover_texts = [
                f"<b>Title: {row['name']}</b><br>"
                f"Song Number: {i + 1}<br>"
                f"Song Start Time: {row['Song Start Time']}<br>"
                f"Tempo (BPM): {round(row['Tempo (BPM)'])}<br>"
                f"Key: {row['Key']}<br>"
                f"Duration (mm:ss): {row['Duration (mm:ss)']}<br>"
                f"Tempo Adjustment Factor: {row['Tempo Adjustment Factor']}"
                for i, (_, row) in enumerate(playlist.iterrows())
            ]
            
            
            # Add a single trace
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines+markers',
                    marker=dict(color=colors, symbol=symbols, size=10),
                    hoverinfo='text',
                    hovertext=hover_texts,
                    name="Analyzed Playlist",
                )
            )
            
        if target_profile is not None:

            # Define default target_profile_kwargs and update with function inputs
            target_profile_kwargs = (
                dict(name="Target Profile", mode="lines", line=dict(color="black")) | (target_profile_kwargs or {})
            )
            if target_profile.ndim == 1:
                fig.add_trace(go.Scatter(y=target_profile, **target_profile_kwargs))
            elif target_profile.ndim == 2:
                fig.add_trace(go.Scatter(x=target_profile[:, 0], y=target_profile[:, 1], **target_profile_kwargs))
            else:
                raise ValueError("Target profile must not be greater than 2D. ")
        
        update_layout_kwargs = (
            dict(
                xaxis_title='Song Index',
                yaxis_title='Tempo (BPM)',
                title='Playlist Tempo Analysis',
                template="plotly_white",
            ) | (update_layout_kwargs or {})
        )
        fig.update_layout(**update_layout_kwargs)
        
        return fig
    
    
    
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
        #>>> validate_variable_types([('data', data, (np.ndarray, NoneType))])
    
        #>>> validate_variable_types([('data', data, (list, NoneType))])
        Traceback (most recent call last):
            ...
        TypeError: data is of type <class 'numpy.ndarray'> when it should be one of (<class 'list'>, <class 'NoneType'>).
    
        #>>> validate_variable_types([('data', data, (list, NoneType), "CUSTOM ERROR MESSAGE HERE.")])
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
                    
########################################################################################################################
# Custom Exception
########################################################################################################################

class SpotifyNotActivatedError(Exception):
    """A custom exception for the DanceDJ class to be raised when someone attempts to use Spotipy functionality after
    disabling Spotipy connectivity."""
    pass