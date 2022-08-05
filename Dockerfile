FROM python:3.10 as venv

RUN python -m venv /pyenv
RUN /pyenv/bin/pip install --no-cache-dir --upgrade pip
RUN /pyenv/bin/pip install --no-cache-dir numpy pandas tsai ipython paho-mqtt influxdb_client

RUN mkdir /app
ENV PATH="/pyenv/bin:${PATH}"

WORKDIR /app
ADD tsai.pkl /app/tsai.pkl
ADD dishwasher.py /app/dishwasher.py
CMD ["/pyenv/bin/python", "/app/dishwasher.py"]
