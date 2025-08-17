labels = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']

LIKELIHOODS = {'acousticness','liveness','instrumentalness'}
CONTINUOUS  = {'speechiness', 'danceability','energy','valence'}

def load_model():
    import vibenet.backends
    return vibenet.backends.EfficientNetModel() # Only model for now