# Graphs/run.py
# Allow running the app.py from outside the interface web directory
import os
import sys

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from web_interface.app import app

if __name__ == '__main__':
    app.run(debug=True)