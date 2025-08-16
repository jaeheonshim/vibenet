# VibeNet: Music Emotion Prediction

A while ago I canceled my Spotify subscription and began building my own offline music library. But something I always missed was Spotify's ability to generate smart playlists like *Morning Mix* or *Driving Vibes*. Spotify doesn't publish how its algorithm creates these playlists, so I set out to build my own system.

As humans, music can make us feel happy, sad, energetic, angry, calm, or a variety of other emotions. Can music make computers *feel* these emotions too? Well, maybe not, but we can teach them to recognize and quantify the qualities of music that drive those feelings. Then, we could use our computers to classify and organize our musical tracks by the emotions they make us feel and recommend songs that match a given mood.

VibeNet is a lightweight Python package and CLI that predicts musical emotions and attributes (valence, energy, danceability, acousticness, etc.) directly from raw audio. It utilizes a distilled EfficientNet student model trained with teacher-student distillation on the [Free Music Archive (FMA) dataset](https://github.com/mdeff/fma).

## What attributes are predicted?

VibeNet predicts 7 continuous attributes for each audio track: **acousticness**, **danceability**, **energy**, **instrumentalness**, **liveness**, **speechiness**, and **valence**.

Some features (like acousticness, instrumentalness, and liveness) are **likelihoods**: they represent probabilities of that characteristic being present (e.g. probability the track is acoustic). Others (like danceability, energy, valence) are **continuous descriptors**: they describe how much of the quality the track has.

For example, an acousticness value of 0.85 doesn't mean that 85% of the track is composed of acoustic instruments. It means that it's highly likely that the recording is acoustic and not electronic.

Conversely, an energy value of 0.15 doesn't mean that it's highly unlikely that the song is energetic. It reflects a degree of the quality itself, meaning that the track is overall perceived as having very low intensity.

Below is a table describing each attribute in more detail:

| Attribute | Type | Description |
|---|---|---|
| **Acousticness** | Likelihood | A measure of how likely a track is to be acoustic rather than electronically produced. High values indicate recordings that rely on natural, unprocessed sound sources (e.g. solo guitar, piano ballad, etc.). Low values indicate tracks that are primarily electronic or produced with synthetic instrumentation (e.g. EDM, trap, etc.) 
| **Instrumentalness** | Likelihood | Predicts whether a track contains no vocals. Higher values suggest that the track contains no vocal content (e.g. symphonies), while lower values indicate the the presence of sung or spoken vocals (e.g. rap).
| **Liveness** | Likelihood | A measure of how likely the track is to be a recording of a live performance. Higher values suggest the presence of live-performance characteristics (crowd noise, reverberation, stage acoustics), while lower values suggest a studio recording.
| **Danceability** | Descriptor | Describes how suitable a track is for dancing. Tracks with higher values (closer to 1.0) tend to feel more danceable while, lower values (closer to 0.0) may feel less danceable.
| **Energy** | Descriptor | Also known as arousal. Measures the perceived intensity and activity level of a track. Higher values indicate tracks that feel fast, loud, and powerful, while lower values indicate tracks that feel calm, soft, or subdued.
| **Valence** | Descriptor | Measures the musical positivity conveyed by a track. Higher values indicate tracks that sound more positive (e.g. happy, cheerful, euphoric), while lower values  indicate tracks that sound more negative (e.g. sad, depressed, angry).
| **Speechiness** | Descriptor | Measures the presence of spoken words in a track. Higher values indicate that the recording is more speech-like (e.g. podcasts), while lower values suggest that the audio is more musical, with singing or purely instrumental content. Mid-range values often correspond to tracks that mix both elements, such as spoken-word poetry layered over music.