# Use the offiical Python base image
FROM python:3.12

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN pip install --no-deps ultralytics

# Copy the rest of the application code
COPY web.py .

# Expose port 8501
EXPOSE 8501

# Set the entry point for the container
CMD ["streamlit", "run", "web.py"]
