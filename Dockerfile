# -----------------------------------------------------------
#  Production inference image — ONNX only, ~200 MB, no CUDA.
#  Build:  docker build -t neu-defect .
#  Run:    docker run --rm -v $(pwd)/samples:/samples \
#              neu-defect python -m src.predict \
#              --onnx neu_defect_vit.onnx --image /samples/test.bmp
# -----------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

COPY src/ ./src/
COPY neu_defect_vit.onnx .

ENTRYPOINT ["python", "-m", "src.predict"]
CMD ["--help"]
