import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../models/detected_sign_preview.dart';
import '../service/model_service.dart';
import '../widgets/detection_overlay.dart';
import '../widgets/recent_signs_strip.dart';
import 'model_settings_screen.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({required this.cameras, super.key});

  final List<CameraDescription> cameras;

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late final CameraController _cameraController;
  late final Future<void> _initializeCameraFuture;
  final ModelService _modelService = ModelService.instance;

  final List<DetectedSignPreview> _recentSigns = [];
  List<DetectedSignPreview> _detections = const [];
  bool _isPredicting = false;
  DateTime _lastPredictionAt = DateTime.fromMillisecondsSinceEpoch(0);

  @override
  void initState() {
    super.initState();
    _cameraController = CameraController(
      widget.cameras.first,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    _initializeCameraFuture = _initializeCamera();
  }

  @override
  void dispose() {
    if (_cameraController.value.isStreamingImages) {
      unawaited(_cameraController.stopImageStream());
    }
    _cameraController.dispose();
    super.dispose();
  }

  Future<void> _initializeCamera() async {
    await _cameraController.initialize();
    await _modelService.initialize();

    if (!_cameraController.value.isStreamingImages) {
      await _cameraController.startImageStream(_handleCameraImage);
    }
  }

  Future<void> _handleCameraImage(CameraImage image) async {
    final now = DateTime.now();
    if (_isPredicting ||
        now.difference(_lastPredictionAt) < const Duration(milliseconds: 250)) {
      return;
    }

    _isPredicting = true;
    _lastPredictionAt = now;
    try {
      final detections = await _modelService.predict(image);
      if (!mounted) {
        return;
      }

      setState(() {
        _detections = detections;
        for (final detection in detections) {
          _recentSigns
            ..removeWhere((recent) => recent.label == detection.label)
            ..insert(0, detection);
        }
        if (_recentSigns.length > 8) {
          _recentSigns.removeRange(8, _recentSigns.length);
        }
      });
    } finally {
      _isPredicting = false;
    }
  }

  void _openModelSettings() {
    Navigator.of(context).push(
      MaterialPageRoute<void>(builder: (_) => const ModelSettingsScreen()),
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
              DetectionOverlay(detections: _detections),
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
