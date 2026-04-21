FROM public.ecr.aws/lambda/python:3.12

# System dependencies for PDF rendering and OpenCV
RUN unset LD_LIBRARY_PATH && dnf install -y poppler-utils mesa-libGL && dnf clean all

# Python dependencies — prefer pre-built wheels, fallback to source build
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN python3.12 -m pip install --no-cache-dir \
    --platform manylinux2014_x86_64 \
    --only-binary=:all: \
    -r ${LAMBDA_TASK_ROOT}/requirements.txt \
    || python3.12 -m pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy handler
COPY handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.lambda_handler"]
