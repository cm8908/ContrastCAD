FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN conda install -c conda-forge pythonocc-core=7.5.1

COPY . .