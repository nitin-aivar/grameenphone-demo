FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.12

# Install system dependency for PDF processing
RUN unset LD_LIBRARY_PATH && dnf install -y poppler-utils mesa-libGL && dnf clean all

# Install Python dependencies — force linux/amd64 platform for pip
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN python3.12 -m pip install --no-cache-dir \
    --platform manylinux2014_x86_64 \
    --only-binary=:all: \
    -r ${LAMBDA_TASK_ROOT}/requirements.txt \
    || python3.12 -m pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy Lambda handler code
COPY handler.py ocr_engine.py json_formatter.py doc_classifier.py dynamo_logger.py prompts.yaml ${LAMBDA_TASK_ROOT}/

CMD ["handler.lambda_handler"]