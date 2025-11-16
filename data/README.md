# Data Folder

This folder is intended to store the video files used for testing the object detection system.

## Important Notice  
Traffic videos are **not included** in this repository due to:

- copyright restrictions,
- large file sizes,
- and GitHub storage limitations.

Users who wish to test the system must provide their own video files and place them in this folder.

## Video Sources (References)

The test videos used during development were downloaded from **Pexels**, a platform that provides free-to-use videos under the Pexels License.

Example categories used:
- Traffic videos  
- Night-time traffic  
- Rainy weather traffic  

Pexels traffic videos can be found here:
https://www.pexels.com/search/videos/traffic/

Users may choose any suitable clip from Pexels or upload their own footage for testing.

## Supported Input  
The detection script accepts:

- MP4 files  
- AVI files  
- MOV files  
- Webcam streams (device index 0 or ID)

Example usage:
```
python src/detect_phase3.py --source data/your_video.mp4
```

Please ensure that the video file path is correct when running the script.
