FROM python:3.13-slim

WORKDIR /app

COPY ./pyproject.toml /app/
COPY ./uv.lock /app/
COPY openai_chat_proxy.py /app/

# RUN apt update && apt install curl -y
# RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# RUN /root/.local/bin/uv sync
# CMD ["/root/.local/bin/uv", "run", "openai_chat_proxy.py"]

RUN pip install uv
RUN uv sync

CMD ["uv", "run", "openai_chat_proxy.py"]