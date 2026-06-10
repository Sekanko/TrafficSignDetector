import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../models/detected_sign_preview.dart';
import '../widgets/recent_signs_strip.dart';
import 'model_settings_screen.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({
    required this.cameras,
    super.key,
  });

  final List<CameraDescription> cameras;

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late final CameraController _cameraController;
  late final Future<void> _initializeCameraFuture;

  final List<DetectedSignPreview> _recentSigns = const [];

  @override
  void initState() {
    super.initState();
    _cameraController = CameraController(
      widget.cameras.first,
      ResolutionPreset.high,
      enableAudio: false,
    );
    _initializeCameraFuture = _cameraController.initialize();
  }

  @override
  void dispose() {
    _cameraController.dispose();
    super.dispose();
  }

  void _openModelSettings() {
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => const ModelSettingsScreen(),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: FutureBuilder<void>(
        future: _initializeCameraFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState != ConnectionState.done) {
            return const Center(child: CircularProgressIndicator());
          }

          return Stack(
            fit: StackFit.expand,
            children: [
              CameraPreview(_cameraController),
              SafeArea(
                child: Align(
                  alignment: Alignment.topRight,
                  child: Padding(
                    padding: const EdgeInsets.only(top: 10, right: 10),
                    child: IconButton.filledTonal(
                      onPressed: _openModelSettings,
                      icon: const Icon(Icons.settings),
                      tooltip: 'Ustawienia modeli',
                    ),
                  ),
                ),
              ),
              SafeArea(
                child: Align(
                  alignment: Alignment.bottomCenter,
                  child: RecentSignsStrip(signs: _recentSigns),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}
