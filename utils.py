from ffmpeg import FFmpeg
from datetime import timedelta
import glob
import os
from mutagen.wave import WAVE

def transcode_to_audio(filename, wavpath):
    ffmpeg = (
        FFmpeg()
        .input(filename)
        .output(wavpath, format="wav")
    )
    ffmpeg.execute()

def writeSrt(chunks):
    result = ""
    idcount = 0
    for segment in chunks:
        if (segment['timestamp'][0] == None):
            continue
        startTime = str(0)+str(timedelta(seconds=int(segment['timestamp'][0])))+',000'
        # failsafe for when end time is null
        if segment['timestamp'][1] == None:
            endTime = str(0)+str(timedelta(seconds=int(segment['timestamp'][0])+5))+',000'
        else:
            endTime = str(0)+str(timedelta(seconds=int(segment['timestamp'][1])))+',000'
        text = segment['text'].strip()
        segmentId = idcount + 1
        idcount += 1
        segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] == ' ' else text}\n\n"
        result += segment
    return result

def filter_files(root_folder, extensions):
    for filenames in glob.glob(root_folder+'/**', recursive=True):
        if filenames.endswith(extensions):
            filename_no_ext = os.path.splitext(filenames)[0]
            srtpath = filename_no_ext + '.en.srt'
            if (os.path.exists(srtpath)):
                continue
            yield filenames

def get_length(filename):
    audio = WAVE(filename)
    return audio.info.length

def round_str(num):
    return str(round(num, 2))