# SelfDrivingCar_DQL

This is my machine learning final project I created with Sergio and Aiden

We used the gymnasium library and a DQL to train the car to navigate the track

## Requirements

- Python 3.11.9
- Microsoft Visual C++ 14.0 or greater (Install from [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/))
- SWIG

## Setup Instructions

1. Install Microsoft Visual C++ Build Tools:
   - Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Run the installer
   - Select "Desktop development with C++" workload
   - In the right panel under "Installation details", ensure these are selected:
     * MSVC C++ x64/x86 build tools
     * Windows 10/11 SDK
     * C++ CMake tools for Windows
   - Click Install

2. Install SWIG:
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
   ```

4. Configure VSCode:
   - Press Ctrl+Shift+P
   - Type "Python: Select Interpreter"
   - Choose the interpreter from the 'venv' directory

5. Run the simulation:
   ```bash
   python simulation_test.py
   ```
