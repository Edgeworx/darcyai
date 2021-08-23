FROM edgeworx/darcy-ai-sdk-base:1.0.0

RUN python3 -m pip install falconn

COPY src/examples/static /src/
COPY src/darcyai/ /src/darcyai
COPY src/people_counting.py /src/demo.py

ENTRYPOINT ["/bin/bash", "-c", "cd /src/ && python3 ./demo.py"]
