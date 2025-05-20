import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';
import 'package:image/image.dart' as img;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  final firstCamera = cameras.first;
  runApp(MyApp(camera: firstCamera));
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;
  const MyApp({required this.camera, Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: HandDetectorScreen(camera: camera),
    );
  }
}

class HandDetectorScreen extends StatefulWidget {
  final CameraDescription camera;
  const HandDetectorScreen({required this.camera, Key? key}) : super(key: key);

  @override
  _HandDetectorScreenState createState() => _HandDetectorScreenState();
}

class _HandDetectorScreenState extends State<HandDetectorScreen> {
  CameraController? _controller;
  Interpreter? _detectionInterpreter;
  Interpreter? _landmarkInterpreter;
  List<Offset> _landmarks = [];
  String _gesture = "None";
  bool _isProcessing = false;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _loadModels();
  }

  Future<void> _initializeCamera() async {
    _controller = CameraController(widget.camera, ResolutionPreset.medium);
    await _controller!.initialize();
    if (!mounted) return;
    _controller!.startImageStream(_processCameraImage);
    setState(() {});
  }

  Future<void> _loadModels() async {
    _detectionInterpreter = await Interpreter.fromAsset('assets/models/hand_detection.tflite');
    _landmarkInterpreter = await Interpreter.fromAsset('assets/models/hand_landmark.tflite');
  }

  void _processCameraImage(CameraImage image) async {
    if (_isProcessing || _detectionInterpreter == null || _landmarkInterpreter == null) return;
    _isProcessing = true;

    // Convert CameraImage to img.Image
    final convertedImage = _convertCameraImage(image);
    final input = _preprocessImage(convertedImage);

    // Step 1: Hand Detection
    final detectionOutput = List.filled(1 * 4, 0.0).reshape([1, 4]); // Example: bounding box output
    _detectionInterpreter!.run(input, detectionOutput);

    // Check if hand is detected
    if (detectionOutput[0][0] < 0.5) { // Simplified threshold check
      setState(() {
        _landmarks = [];
        _gesture = "None";
      });
      _isProcessing = false;
      return;
    }

    // Step 2: Landmark Estimation
    final landmarkInput = _cropAndPreprocessForLandmarks(convertedImage, detectionOutput.cast<List<double>>());
    final landmarkOutput = List.filled(21 * 3, 0.0).reshape([1, 21, 3]); // 21 landmarks, 3D coords
    _landmarkInterpreter!.run(landmarkInput, landmarkOutput);

    // Extract landmarks
    _landmarks = _processLandmarkOutput(landmarkOutput.cast<List<List<double>>>());
    _gesture = _recognizeGesture(_landmarks);

    setState(() {});
    _isProcessing = false;
  }

  img.Image _convertCameraImage(CameraImage image) {
    // Simplified YUV to RGB conversion
    final width = image.width;
    final height = image.height;
    final yPlane = image.planes[0].bytes;
    final uPlane = image.planes[1].bytes;
    final vPlane = image.planes[2].bytes;

    final converted = img.Image(width, height);
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final yIdx = y * width + x;
        final uIdx = (y ~/ 2) * (width ~/ 2) + (x ~/ 2);
        final vIdx = uIdx;

        final Y = yPlane[yIdx];
        final U = uPlane[uIdx] - 128;
        final V = vPlane[vIdx] - 128;

        final r = (Y + 1.13983 * V).clamp(0, 255).toInt();
        final g = (Y - 0.39465 * U - 0.58060 * V).clamp(0, 255).toInt();
        final b = (Y + 2.03211 * U).clamp(0, 255).toInt();

        converted.setPixelRgba(x, y, r, g, b, 255);
      }
    }
    return converted;
  }

 TensorBuffer _preprocessImage(img.Image image) {
  final resized = img.copyResize(image, width: 192, height: 192);
  final input = TensorBuffer.createFixedSize([1, 192, 192, 3], TfLiteType.float32);
  for (var y = 0; y < 192; y++) {
    for (var x = 0; x < 192; x++) {
      final pixel = resized.getPixel(x, y); // Returns an int
      final r = img.getRed(pixel).toDouble();
      final g = img.getGreen(pixel).toDouble();
      final b = img.getBlue(pixel).toDouble();
      input.getBuffer().asFloat32List()[y * 192 * 3 + x * 3 + 0] = r / 255.0;
      input.getBuffer().asFloat32List()[y * 192 * 3 + x * 3 + 1] = g / 255.0;
      input.getBuffer().asFloat32List()[y * 192 * 3 + x * 3 + 2] = b / 255.0;
    }
  }
  return input;
}

  TensorBuffer _cropAndPreprocessForLandmarks(img.Image image, List<List<double>> detectionOutput) {
  // Simplified: Crop image based on detection bounding box
  // In practice, use detectionOutput to crop the hand region
  final cropped = img.copyResize(image, width: 224, height: 224); // Adjust based on actual bbox
  final input = TensorBuffer.createFixedSize([1, 224, 224, 3], TfLiteType.float32);
  for (var y = 0; y < 224; y++) {
    for (var x = 0; x < 224; x++) {
      final pixel = cropped.getPixel(x, y); // Returns an int
      final r = img.getRed(pixel).toDouble();
      final g = img.getGreen(pixel).toDouble();
      final b = img.getBlue(pixel).toDouble();
      input.getBuffer().asFloat32List()[y * 224 * 3 + x * 3 + 0] = r / 255.0;
      input.getBuffer().asFloat32List()[y * 224 * 3 + x * 3 + 1] = g / 255.0;
      input.getBuffer().asFloat32List()[y * 224 * 3 + x * 3 + 2] = b / 255.0;
    }
  }
  return input;
}

  List<Offset> _processLandmarkOutput(List<List<List<double>>> output) {
    final landmarks = <Offset>[];
    for (var point in output[0]) {
      final x = point[0] * 224; // Scale to display size
      final y = point[1] * 224;
      landmarks.add(Offset(x, y));
    }
    return landmarks;
  }

  String _recognizeGesture(List<Offset> landmarks) {
    if (landmarks.length < 21) return "UNKNOWN";

    final fingers = <int>[];
    final tipIds = [4, 8, 12, 16, 20]; // Thumb, Index, Middle, Ring, Pinky

    // Thumb detection
    final thumbTip = landmarks[tipIds[0]];
    final thumbIp = landmarks[tipIds[0] - 1];
    fingers.add((thumbTip.dx - thumbIp.dx).abs() > 0.05 * 224 ? 1 : 0);

    // Other fingers detection
    for (var i = 1; i < 5; i++) {
      final fingerTip = landmarks[tipIds[i]];
      final fingerDip = landmarks[tipIds[i] - 2];
      fingers.add(fingerTip.dy < fingerDip.dy ? 1 : 0);
    }

    final fingerBinary = fingers.join();

    // Gesture dictionary from handpromax.py
    const gestures = {
      "11111": "hello",
      "00000": "goodbye",
      "01000": "POINTING",
      "00011": "OK",
      "01111": "THUMBS UP",
      "11110": "THUMBS DOWN",
      "01011": "ROCK ON",
      "11001": "LOVE",
      "00111": "PEACE",
    };

    return gestures[fingerBinary] ?? "UNKNOWN";
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }
    return Stack(
      children: [
        CameraPreview(_controller!),
        CustomPaint(
          painter: LandmarkPainter(_landmarks),
          child: Container(),
        ),
        Positioned(
          bottom: 30,
          left: MediaQuery.of(context).size.width / 2 - 50,
          child: Text(
            "Gesture: $_gesture",
            style: const TextStyle(color: Colors.green, fontSize: 20, backgroundColor: Colors.black54),
          ),
        ),
        const Positioned(
          top: 20,
          left: 20,
          child: Text(
            "Show hand gestures to camera",
            style: TextStyle(color: Colors.white, fontSize: 16, backgroundColor: Colors.black54),
          ),
        ),
      ],
    );
  }

  @override
  void dispose() {
    _controller?.dispose();
    _detectionInterpreter?.close();
    _landmarkInterpreter?.close();
    super.dispose();
  }
}

class LandmarkPainter extends CustomPainter {
  final List<Offset> landmarks;
  LandmarkPainter(this.landmarks);

  @override
  void paint(Canvas canvas, Size size) {
    final pointPaint = Paint()
      ..color = const Color.fromRGBO(121, 22, 76, 1.0)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.fill;

    final linePaint = Paint()
      ..color = const Color.fromRGBO(250, 44, 250, 1.0)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    // Draw landmarks
    for (var landmark in landmarks) {
      canvas.drawCircle(landmark, 2.0, pointPaint);
    }

    // Draw connections (simplified, based on MediaPipe hand connections)
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
      [0, 5], [5, 6], [6, 7], [7, 8], // Index
      [0, 9], [9, 10], [10, 11], [11, 12], // Middle
      [0, 13], [13, 14], [14, 15], [15, 16], // Ring
      [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
    ];

    for (var conn in connections) {
      if (landmarks.length > conn[1]) {
        canvas.drawLine(landmarks[conn[0]], landmarks[conn[1]], linePaint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}