import os
from tqdm import tqdm
from timeit import default_timer as timer
import utils
import transcribe
import config

# ignore warnings
import warnings
import logging
warnings.filterwarnings("ignore", ".*1Torch was not compiled with flash attention.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Get the number of wav files in the root folder and its sub-folders
print("Getting number of files to transcribe...")
extensions = (".mp4",".wmv",".mov",".avi",".mpg", ".mkv")
files_list = []
for folder in config.scan_folders:
    files_list += list(utils.filter_files(folder, extensions))
print("Number of files: ", len(files_list))

# Transcribe the wav files and display a progress bar
with tqdm(total=len(files_list), desc="Transcribing Files") as pbar:
    for filename in files_list:
        filename_no_ext = os.path.splitext(filename)[0]
        file_str = filename_no_ext.rsplit("\\", 1)[1]
        pbar.set_postfix_str(file_str)
        #wavpath = "F:/whisper-tmp.wav"
        wavpath = filename_no_ext + '.wav'
        try:
            transcode_start = timer()
            utils.transcode_to_audio(filename, wavpath)
            transcode_end = timer()
            transcribe_start = timer()
            transcribe.transcribe_file(wavpath, filename_no_ext)
            transcribe_end = timer()
            filelen = utils.get_length(wavpath)
            os.remove(wavpath)
            msg = "Ln | " + str(round(filelen, 2)) + " | Tx " + str(round(transcode_end - transcode_start, 2)) + " | Ts " + str(round(transcribe_end - transcribe_start, 2)) + " | Multi " + str(round(filelen/(transcode_end - transcode_start), 2)) + "x"
            pbar.write(msg)
        except Exception as e:
            pbar.write("Error: " + str(e) + " " + file_str)
        pbar.update(1)