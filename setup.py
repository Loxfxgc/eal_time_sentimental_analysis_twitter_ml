import nltk
import subprocess
import sys
from setuptools import setup, find_packages

# Download NLTK stopwords
print("Downloading NLTK stopwords...")
try:
    nltk.download('stopwords')
    print("NLTK stopwords downloaded successfully.")
except Exception as e:
    print(f"Error downloading NLTK stopwords: {e}")
    print("You may need to download stopwords manually with: python -m nltk.downloader stopwords")

# Install requirements
print("Installing requirements...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Requirements installed successfully.")
except Exception as e:
    print(f"Error installing requirements: {e}")

setup(
    name="twitter_sentiment_analysis",
    version="0.1",
    packages=find_packages(),
) 