FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ninja-build \
    texlive-latex-base texlive-fonts-recommended texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp (CPU only) for its inference tools. No architecture patch is
# needed here -- that patch fixed *producing* GLM-OCR GGUF files; the ones this
# Space downloads (ctogaurav/GLM_OCR-GGUF) are already fixed.
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp \
    && cd /opt/llama.cpp \
    && cmake -B build -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -G Ninja \
    && cmake --build build --config Release --target llama-mtmd-cli -j \
    && cp build/bin/llama-mtmd-cli /usr/local/bin/ \
    && cp build/bin/*.so /usr/local/lib/ 2>/dev/null || true \
    && ldconfig \
    && rm -rf /opt/llama.cpp/build/CMakeFiles

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# HF Spaces expects the app on 7860
ENV GRADIO_SERVER_PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
