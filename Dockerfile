# Use the official Python image as the base image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Install gcc and other necessary build tools
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    libffi-dev \
    libheif-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python libraries
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pydantic

# Copy the FastAPI app code into the container
COPY . .

# Expose the port that the FastAPI app runs on
EXPOSE 3059

# Define the command to run the FastAPI app using uvicorn
CMD ["gunicorn", "api:app", "--host", "0.0.0.0", "--port", "3059"]
