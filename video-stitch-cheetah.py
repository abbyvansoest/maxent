import os
import glob
from moviepy.editor import VideoFileClip, concatenate_videoclips

run_dir = 'cheetah/videos/reduce4_proper_dims_recorded/'
subdirs = ['entropy/', 'mixed/', 'baseline/']

for sub in subdirs:
    base_dir = run_dir+sub
    clips = []
    epochs = sorted([f for f in os.listdir(base_dir) if not f.startswith('.')])
    for c in epochs:
        c = c + '/'
        print(base_dir+c)
        epoch_clips = []

        # for rollout in os.listdir(base_dir + c):
        mp4s = glob.glob(base_dir + c + '/*.mp4')
        epoch_clips = epoch_clips + [VideoFileClip(mp4) for mp4 in mp4s]
        
        # save the clips from this epoch
        # final_epoch_clip = concatenate_videoclips(epoch_clips)
        # final_epoch_clip.write_videofile(base_dir+c+'video.mp4')
        clips += epoch_clips

    final_exp_clip = concatenate_videoclips(clips)
    final_exp_clip.write_videofile(base_dir+'video.mp4')
