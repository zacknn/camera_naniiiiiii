# Hand Gesture Recognition App

This is a Flutter application that uses MediaPipe's hand detection and landmark models to recognize hand gestures in real-time via the phone's camera. The app displays landmarks on the hand and identifies gestures such as "PEACE", "OK", "THUMBS UP", and more.

## Features
- Opens the phone camera to capture live video.
- Runs MediaPipe's `hand_detection.tflite` and `hand_landmark.tflite` models.
- Extracts and displays 21 hand landmarks in real-time.
- Recognizes 9 predefined gestures: "hello", "goodbye", "POINTING", "OK", "THUMBS UP", "THUMBS DOWN", "ROCK ON", "LOVE", "PEACE".
- Mirrors the camera feed for a natural user experience.

## Prerequisites
- **Flutter SDK**: Installed and configured (check with `flutter doctor`).
- **Android SDK**: Installed with NDK (e.g., version 26.3.11579264 or later).
- **Device**: A physical Android phone (e.g., Samsung Galaxy A21s) with a working camera.
- **Internet**: Required to download dependencies and models initially.

- 2. Install Dependencies

    Ensure pubspec.yaml is updated with the following dependencies:
    yaml
'''bash
## dependencies:
  # flutter:
    sdk: flutter
  camera: ^0.10.5+9
  tflite_flutter: ^0.10.4
  tflite_flutter_helper: ^0.3.1
  image: ^4.2.0
'''
Run:
bash

    flutter pub get

3. Add TFLite Models

    Download the MediaPipe Hand Landmarker model from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
    Extract hand_detection.tflite and hand_landmark.tflite from the .task file (use unzip or a tool like 7-Zip).
    Place them in the assets/models/ directory.
    Update pubspec.yaml with:
    yaml

    flutter:
      assets:
        - assets/models/hand_detection.tflite
        - assets/models/hand_landmark.tflite

4. Configure Permissions

    Android: Add the following to android/app/src/main/AndroidManifest.xml:
    xml

<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-feature android:name="android.hardware.camera" android:required="true" />
<uses-feature android:name="android.hardware.microphone" android:required="true" />
iOS: Add the following to ios/Runner/Info.plist:
xml

    <key>NSCameraUsageDescription</key>
    <string>App needs camera access to detect hand gestures</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>App needs microphone access for potential voice features</string>

5. Build and Run

    Connect a physical Android device via USB and enable Developer Mode with USB Debugging.
    Run:
    bash

    flutter clean
    flutter pub get
    flutter run
    Select the connected device if prompted.

Testing Instructions

    Launch the App:
        The app should open and display a live camera feed.
    Hand Detection:
        Place your hand in front of the camera. Red dots (landmarks) and purple lines (connections) should appear on your hand.
    Gesture Recognition:
        Perform gestures like:
            "PEACE": Index and middle fingers up.
            "OK": Thumb and index forming a circle.
            "THUMBS UP": Thumb up, other fingers closed.
        Check the bottom center of the screen for the recognized gesture (e.g., "Gesture: PEACE").
    Troubleshooting:
        If no landmarks appear, verify the model files are correct and in the right directory.
        If gestures show "UNKNOWN", try adjusting hand position or lighting.
        Check the console for errors if the app crashes.

Known Issues

    The app may lag on low-end devices due to real-time processing.
    Gesture recognition accuracy depends on lighting and hand orientation.
    NDK configuration issues may require reinstalling the Android NDK (see Prerequisites).

Contributing

Feel free to report issues or suggest enhancements. Contact the developer for support.
