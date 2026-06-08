FROM lmsysorg/sglang-omni:dev

WORKDIR /app

RUN git clone https://github.com/sgl-project/sglang-omni.git && \
    cd sglang-omni && \
    uv venv .venv -p 3.12 && \
    /bin/zsh -c "source .venv/bin/activate && uv pip install -v -e ."

ENV PATH="/app/sglang-omni/.venv/bin:$PATH"
