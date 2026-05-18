FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY server /app/server
COPY sdk-python /app/sdk-python
RUN pip install ./server ./sdk-python \
 && pip install 'uvicorn[standard]' scikit-learn
EXPOSE 7187
CMD ["uvicorn", "langpred_server.main:app", "--host", "0.0.0.0", "--port", "7187"]
