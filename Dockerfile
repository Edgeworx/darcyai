FROM edgeworx/darcy-ai-sdk-base:1.0.0

RUN python3 -m pip install falconn

COPY src/examples/*.tflite /src/
COPY src/examples/*_labels.txt /src/
COPY src/darcyai/ /src/darcyai
COPY src/examples/static/ /src/static/
COPY src/examples/people_perception.py /src/app.py

ENTRYPOINT ["/bin/bash", "-c", "cd /src/ && python3 ./app.py"]
