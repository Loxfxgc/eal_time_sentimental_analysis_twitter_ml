import nltk
import os
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Create NLTK data directory if it doesn't exist
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Download stopwords
print("Downloading NLTK stopwords...")
try:
    nltk.download('stopwords')
    print("NLTK stopwords downloaded successfully to:", nltk_data_dir)
except Exception as e:
    print(f"Error downloading NLTK stopwords: {e}")

# Verify the download
try:
    from nltk.corpus import stopwords
    words = stopwords.words('english')
    print(f"Successfully loaded {len(words)} stopwords!")
    print("Sample stopwords:", words[:5])
except Exception as e:
    print(f"Error verifying stopwords: {e}") 