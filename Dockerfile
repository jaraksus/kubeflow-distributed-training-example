FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

WORKDIR /work

COPY data /work/data

COPY main.py /work/
COPY trainer.py /work/
COPY dataset.py /work/

ENTRYPOINT [ "python3", "-u", "./main.py" ]
