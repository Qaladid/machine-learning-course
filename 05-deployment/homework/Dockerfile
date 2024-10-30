# Start from the prepared base image
FROM svizor/zoomcamp-model:3.11.5-slim

# Set the working directory
WORKDIR /app

# # Install pipenv (if needed)
RUN pip install pipenv

# # Copy Pipfile and Pipfile.lock
COPY ["Pipfile", "Pipfile.lock", "./"]

# # Install dependencies
RUN pipenv install --system --deploy

# # Copy your application files
COPY ["predict1.py", "model1.bin", "dv.bin", "./"]

# # Expose the port
EXPOSE 9696

# # Run the application with Gunicorn
ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]
