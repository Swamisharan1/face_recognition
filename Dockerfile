FROM python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Set a working directory
WORKDIR /app

# Copy your Streamlit app files to the container
COPY . .

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Expose the port your Streamlit app is running on
EXPOSE 8501

# Start your Streamlit app
CMD ["streamlit", "run", "vis_model.py"]
