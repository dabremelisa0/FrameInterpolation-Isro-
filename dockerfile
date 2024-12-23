# Use an official Python runtime as a base image
FROM python:3.11.7-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application directory including the film_model folder
COPY . .

# If you prefer to be explicit, you could do:
# COPY film_model/ film_model/

# Set the PORT environment variable to 8080 (Cloud Run uses this)
ENV PORT=8080

# Expose the port on which the app runs
EXPOSE 8080

# Command to run the FastAPI app using Python
CMD ["python", "main.py"]
