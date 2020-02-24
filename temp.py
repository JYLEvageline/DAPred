vid_candidates = self.get_vids_candidate(int(vids_long[idx][i]),int(vids_long[idx][i+1]))
dist = np.array((len(vid_candidates),length))
for j, vid_candidate in enumerate(vid_candidates):
    for k in range(length):
        dist[j][k] = float(np.exp(-self.get_distance(vid_candidates[j], vids_long[idx][k])))
temp_atten = atten_scores[n+i,:]
score = dist * temp_atten