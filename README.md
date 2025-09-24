# 📸 Countdown Camera Timer

A Streamlit app that shows a countdown timer from 5 and automatically takes a picture when it reaches zero.

## Features

- ⏰ 5-second countdown timer
- 📸 Automatic camera capture when timer reaches zero
- 💾 Saves photos with timestamp
- 🔄 Option to take another photo
- 📱 Responsive design

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. Click "Start Countdown" to begin the 5-second timer
2. Get ready for your photo!
3. The app will automatically capture your image when the countdown reaches zero
4. View your photo and click "Take Another Photo" to repeat

## Requirements

- Python 3.7+
- Webcam/camera access
- Streamlit
- OpenCV
- NumPy

## Notes

- Photos are saved in the `photos/` directory with timestamps
- Make sure to allow camera access when prompted by your browser
- The app works best in Chrome, Firefox, or Safari
