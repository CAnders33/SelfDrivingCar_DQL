# SelfDrivingCar_DQL

This is my machine learning final project I created with Sergio and Aiden

We used the gymnasium library and a DQL to train the car to navigate the track

## Setup Instructions

1. First, install SWIG:
   - Download SWIG from http://www.swig.org/download.html
   - Add SWIG to your system PATH

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install "gymnasium[box2d]"
   ```

4. Configure VSCode:
   - Press Ctrl+Shift+P
   - Type "Python: Select Interpreter"
   - Choose the interpreter from the 'venv' directory

5. Run the simulation:
   ```bash
   python simulation_test.py
   ```
