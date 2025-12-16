# Dockerfile

# Use a specific Python base image for stability
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current directory (your project root) into the container
COPY . /app

# Install all dependencies from requirements.txt
# This must be run before the rest of the application is set upRUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --trusted-host pypi.python.org --trusted-host pypi.org --trusted-host files.pythonhosted.org

# The MLflow database (mlruns.db) must be created/present when the container starts
# The api.py script handles loading the model from this database.

# Expose the port the Flask app will run on
EXPOSE 5000

# Command to run the API when the container starts
# This executes the api.py script
CMD ["python", "api.py"]

RUN pip install --no-cache-dir -r requirements.txt