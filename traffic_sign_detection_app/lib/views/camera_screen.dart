import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../models/detected_sign_preview.dart';
import '../service/model_service.dart';
import '../widgets/detection_overlay.dart';
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

  List<DetectedSignPreview> _detections = const [];
  bool _isPredicting = false;
  DateTime _lastPredictionAt = DateTime.fromMillisecondsSinceEpoch(0);

  @override
  void initState() {
    super.initState();
    _cameraController = CameraController(
      widget.cameras.first,
      ResolutionPreset.low,
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
        now.difference(_lastPredictionAt) < const Duration(seconds: 1)) {
      return;
    }

    _isPredicting = true;
    _lastPredictionAt = now;
    try {
      final detections = await _modelService.predict(
        image,
        rotationDegrees: _cameraController.description.sensorOrientation,
      );
      if (!mounted) {
        return;
      }
      if (detections.isEmpty && _detections.isEmpty) {
        return;
      }

      setState(() {
        _detections = detections;
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

          return ListenableBuilder(
            listenable: _modelService,
            builder: (context, _) {
              final currentModel = _modelService.currentModel;
              final hasLoadedModel = _modelService.hasLoadedModel;
              final modelError = _modelService.lastError;

              return Stack(
                fit: StackFit.expand,
                children: [
                  CameraPreview(_cameraController),
                  DetectionOverlay(detections: _detections),
                  SafeArea(
                    child: Align(
                      alignment: Alignment.topLeft,
                      child: Padding(
                        padding: const EdgeInsets.only(
                          top: 10,
                          left: 10,
                          right: 56,
                        ),
                        child: _ModelStatusBanner(
                          currentModelName: currentModel?.name,
                          hasLoadedModel: hasLoadedModel,
                          error: modelError,
                        ),
                      ),
                    ),
                  ),
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
                ],
              );
            },
          );
        },
      ),
    );
  }
}

class _ModelStatusBanner extends StatelessWidget {
  const _ModelStatusBanner({
    required this.currentModelName,
    required this.hasLoadedModel,
    required this.error,
  });

  final String? currentModelName;
  final bool hasLoadedModel;
  final String? error;

  @override
  Widget build(BuildContext context) {
    final text = hasLoadedModel
        ? 'Aktywny model: $currentModelName'
        : error ?? 'Brak aktywnego modelu. Otwórz ustawienia i pobierz model.';

    return DecoratedBox(
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.72),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        child: Text(
          text,
          style: Theme.of(context).textTheme.bodySmall?.copyWith(
            color: Colors.white,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
    );
  }
}
