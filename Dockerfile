FROM nvcr.io/nvidia/pytorch:26.03-py3
WORKDIR /opt/edgerock-runtime
COPY requirements.txt .
COPY wheels/ ./wheels/
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN CAUSAL_CONV1D_FORCE_BUILD=TRUE python -m pip install --no-cache-dir --no-build-isolation -r requirements.txt
COPY . .
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation .
ENTRYPOINT ["python", "-I", "/opt/edgerock-runtime/run.py"]
