import os
import glob
from moviepy.editor import VideoFileClip, concatenate_videoclips

run_dir = 'humanoid/videos/record_multifolder_5000_take2/'
subdirs = ['baseline/', 'entropy/', 'mixed/']

for sub in subdirs:
    base_dir = run_dir+sub
    clips = []
    for c in sorted(os.listdir(base_dir)):
        c = c + '/'
        epoch_clips = []
        for rollout in os.listdir(base_dir + c):
            mp4s = glob.glob(base_dir + c + rollout + '/*.mp4')
            clips = clips + [VideoFileClip(mp4) for mp4 in mp4s]
            epoch_clips = epoch_clips + [VideoFileClip(mp4) for mp4 in mp4s]
        
        # save the clips from this epoch
        final_epoch_clip = concatenate_videoclips(epoch_clips)
        final_epoch_clip.write_videofile(base_dir+c+'video.mp4')

    final_exp_clip = concatenate_videoclips(clips)
    final_exp_clip.write_videofile(base_dir+'video.mp4')
