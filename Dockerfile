ARG TORCH_CU_VERSION=11.8 \
    TORCH_VER=2.3.0

#pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime
FROM pytorch/pytorch:${TORCH_VER}-cuda${TORCH_CU_VERSION}-cudnn8-runtime
ARG FLASH_VER=2.5.8 \
    FLASH_CU_VERSION=118 \
    FLASH_TORCH_VER=2.3 \
    FLASH_PY_VER=cp310

RUN apt update && \
    apt install ffmpeg wget -y
RUN pip install transformers accelerate datasets[audio]
RUN wget "https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_VER}/flash_attn-${FLASH_VER}+cu${FLASH_CU_VERSION}torch${FLASH_TORCH_VER}cxx11abiTRUE-${FLASH_PY_VER}-${FLASH_PY_VER}-linux_x86_64.whl"
RUN pip install flash_attn-${FLASH_VER}+cu${FLASH_CU_VERSION}torch${FLASH_TORCH_VER}cxx11abiTRUE-${FLASH_PY_VER}-${FLASH_PY_VER}-linux_x86_64.whl
RUN pip install python-ffmpeg mutagen tqdm

COPY . ./
RUN python whisper.py
CMD ["python", "whisper.py"]