import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "ind/data/00_tracks.csv"

# recordingId,trackId,frame,trackLifetime,xCenter,yCenter,heading,width,length,xVelocity,yVelocity,xAcceleration,yAcceleration,lonVelocity,latVelocity,lonAcceleration,latAcceleration
tracks = pd.read_csv(path, header=0, sep=",")
tracks = np.array(tracks)

# trackId, trackLifetime, xCenter, yCenter, xVelocity, yVelocity, heading
tracks_data = np.array(tracks[:, [1, 3, 4, 5, 9, 10, 6]])

plt.figure()
for i in range(int(tracks_data[:, 0].max() + 1)):
    track_temp = tracks_data[tracks_data[:, 0] == i]
    track_temp = track_temp[:, [1, 2, 3, 4, 5, 6]]
    path_npy = "ind/00/" + str(i) + ".npy"
    np.save(path_npy, track_temp)
