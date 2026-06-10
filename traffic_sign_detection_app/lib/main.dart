import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'views/camera_screen.dart';

late final List<CameraDescription> _cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  _cameras = await availableCameras();
  runApp(TrafficSignDetectionApp(cameras: _cameras));
}

class TrafficSignDetectionApp extends StatelessWidget {
  const TrafficSignDetectionApp({
    required this.cameras,
    super.key,
  });

  final List<CameraDescription> cameras;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blue,
          brightness: Brightness.dark,
        ),
        scaffoldBackgroundColor: const Color(0xFF0B0F14),
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blue,
          brightness: Brightness.dark,
        ),
        scaffoldBackgroundColor: const Color(0xFF0B0F14),
        useMaterial3: true,
      ),
      themeMode: ThemeMode.dark,
      home: CameraScreen(cameras: cameras),
    );
  }
}
