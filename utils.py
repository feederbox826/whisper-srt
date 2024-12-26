import ffmpeg
import glob
import os
from mutagen.aac import AAC

def transcode_to_audio(filename, wavpath):
    ffmpeg.input(filename).output(wavpath, acodec="aac").overwrite_output().run(quiet=True)

def filter_files(root_folder, extensions):
    for filenames in glob.glob(root_folder+'/**', recursive=True):
        if filenames.endswith(extensions):
            filename_no_ext = os.path.splitext(filenames)[0]
            srtpath = filename_no_ext + '.en.srt'
            if (os.path.exists(srtpath)):
                continue
            yield filenames

def get_length(filename):
    audio = AAC(filename)
    return audio.info.length