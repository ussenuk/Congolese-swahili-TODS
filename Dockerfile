# Docker file for the Haki Bot
FROM python:3.9
WORKDIR /bot-haki

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
# RUN cd bot-haki

EXPOSE 5005
RUN chmod +rwx ./start_rasa.sh

CMD ["bash", "./start_rasa.sh"]
