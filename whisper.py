# ignore warnings
import warnings
warnings.filterwarnings("ignore", ".*")
warnings.filterwarnings("ignore", "Lightning automatically upgraded your loaded checkpoint.*")
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`.*")
warnings.filterwarnings("ignore", "Model was trained with pyannote.audio 0.0.1.*")
warnings.filterwarnings("ignore", "Model was trained with torch 1.10.0+cu102.+")

from tqdm import tqdm
from timeit import default_timer as timer
import utils
import os
import config
import random

# setup subbing
import whisperx # type: ignore
from whisperx.utils import WriteSRT # type: ignore

device = "cuda" 
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

model_dir = "models"

model = whisperx.load_model("large-v3-turbo", device, language="en", compute_type=compute_type, download_root=model_dir)
model_a, metadata = whisperx.load_align_model(language_code="en", device=device, model_dir=model_dir)

# Get the number of wav files in the root folder and its sub-folders
print("Getting number of files to transcribe...")
extensions = (".mp4",".wmv",".mov",".avi",".mpg", ".mkv")
files_list = []
for folder in config.scan_folders:
    files_list += list(utils.filter_files(folder, extensions))
# if desired, shuffle
if config.shuffle_files:
    random.shuffle(files_list)
files_list_len = len(files_list)
print("Number of files: ", files_list_len)

def transcribe_audio(filename, filename_no_ext=None):
    audio = whisperx.load_audio(filename)
    result = model.transcribe(audio, chunk_size=10, batch_size=batch_size)
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    aligned_result["language"] = "en"
    with open(filename_no_ext + '.en.srt', 'w') as srtfile:
        writesrt = WriteSRT(".")
        writesrt.write_result(aligned_result, srtfile, {"max_line_width": None, "max_line_count": 2, "highlight_words": False, "preserve_segments": True})

# Transcribe the wav files and display a progress bar
with tqdm(total=files_list_len, desc="Transcribing Files") as pbar:
    for filename in files_list:
        filename_no_ext = os.path.splitext(filename)[0]
        file_str = filename_no_ext.rsplit("\\", 1)[1].encode('ascii', 'ignore').decode('ascii')
        pbar.set_postfix_str(file_str)
        # F is a ramdisk, change to preferred path
        wavpath = f"F:\{file_str}.aac" if os.path.exists("F:") else filename_no_ext + '.aac'
        try:
            # create progess bar for in-file processing
            filebar = tqdm(total=2, desc="Processing", leave=False)
            # transcoding (TX)
            transcode_start = timer()
            filebar.update(1)
            filebar.set_description("Transcoding")
            utils.transcode_to_audio(filename, wavpath)
            transcode_end = timer()
            # transcribing (TS)
            transcribe_start = timer()
            filebar.update(1)
            filebar.set_description("Transcribing")
            transcribe_audio(wavpath, filename_no_ext)
            transcribe_end = timer()
            # getting metadata, cleaning up
            filelen = utils.get_length(wavpath)
            os.unlink(wavpath) # delete file
            file_len = round(filelen, 2)
            tx_time = round(transcode_end - transcode_start, 2)
            txx = round(file_len/tx_time, 2)
            ts_time = round(transcribe_end - transcribe_start, 2)
            tsx = round(file_len/ts_time, 2)
            msg = f"Ln | {file_len} | Tx {tx_time} ({txx}x) | Ts {ts_time} ({tsx}x)"
            pbar.write(msg)
        except Exception as e:
            pbar.write("Error: " + str(e) + " " + file_str)
        pbar.update(1)