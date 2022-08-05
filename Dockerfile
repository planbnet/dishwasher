FROM planbnet/tsai
RUN mkdir /app
WORKDIR /app
ADD tsai.pkl /app/tsai.pkl
ADD dishwasher.py /app/dishwasher.py
CMD ["python", "/app/dishwasher.py"]
