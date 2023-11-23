# first line: 36
    @memory.cache
    def analyze_songs(song_id, song_name):
        return dj.analyze_songs([song_id], [song_name], progress_bar=False)
