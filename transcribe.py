import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import utils

import warnings
warnings.filterwarnings("ignore", ".*1Torch was not compiled with flash attention.*")
warnings.filterwarnings("ignore", ".*Whisper did not predict an ending timestamp.*")

# set up whisper
device = "cuda:0"
torch_dtype = torch.float16
model_id = "distil-whisper/distil-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=10,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

def transcribe_file(filename, filename_no_ext=None):
    result = pipe(filename, return_timestamps=True)
    transcription = result['chunks']
    # Write to srt file
    with open(filename_no_ext + '.en.srt', 'w') as srtfile:
        srtresult = utils.writeSrt(transcription)
        srtfile.write(srtresult)