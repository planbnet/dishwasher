# dishwasher

This script won't be very useful to anyone but me. It uses a neural network to classify power data from my home automation system (stored in influxdb) to detect if the dishwasher has finished. The model is trained in a jupyter notebook that is part of a private repo because it also contains other private data.

The Docker image planbnet/tsai might be useful for people trying to run tsai time series models on Python 3.10 in Docker. See Dockerfile for example on how to use it.
