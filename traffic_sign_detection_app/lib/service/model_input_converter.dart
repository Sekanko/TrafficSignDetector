import 'dart:math' as math;
import 'dart:ui';

import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class ModelInputConverter {
  const ModelInputConverter._();

  static Object? cameraImageToInput({
    required CameraImage image,
    required List<int> inputShape,
    required int fallbackSize,
    required TensorType type,
    int rotationDegrees = 0,
  }) {
    final width = inputShape.length >= 3 ? inputShape[2] : fallbackSize;
    final height = inputShape.length >= 3 ? inputShape[1] : fallbackSize;
    final channels = inputShape.length >= 4 ? inputShape[3] : 3;

    if (channels != 3) {
      return null;
    }

    return _cameraImageToRgb(
      image: image,
      width: width,
      height: height,
      normalize: _isFloat(type),
      rotationDegrees: rotationDegrees,
    ).reshape(inputShape);
  }

  static Object? cameraImageCropToInput({
    required CameraImage image,
    required Rect crop,
    required List<int> inputShape,
    required int fallbackSize,
    required TensorType type,
    int rotationDegrees = 0,
  }) {
    final width = inputShape.length >= 3 ? inputShape[2] : fallbackSize;
    final height = inputShape.length >= 3 ? inputShape[1] : fallbackSize;
    final channels = inputShape.length >= 4 ? inputShape[3] : 3;

    if (channels != 3) {
      return null;
    }

    return _cameraImageToRgb(
      image: image,
      width: width,
      height: height,
      normalize: _isFloat(type),
      crop: crop,
      rotationDegrees: rotationDegrees,
    ).reshape(inputShape);
  }

  static Object emptyTensorData(List<int> shape, TensorType type) {
    final elements = shape.fold<int>(1, (total, value) => total * value);
    final data = _isFloat(type)
        ? List<double>.filled(elements, 0)
        : List<int>.filled(elements, 0);

    return data.reshape(shape);
  }

  static bool _isFloat(TensorType type) {
    return type == TensorType.float32 ||
        type == TensorType.float16 ||
        type == TensorType.float64;
  }

  static List<num> _cameraImageToRgb({
    required CameraImage image,
    required int width,
    required int height,
    required bool normalize,
    Rect crop = const Rect.fromLTRB(0, 0, 1, 1),
    int rotationDegrees = 0,
  }) {
    final input = List<num>.filled(width * height * 3, 0);
    var inputIndex = 0;
    final cropLeft = crop.left.clamp(0.0, 1.0);
    final cropTop = crop.top.clamp(0.0, 1.0);
    final cropRight = crop.right.clamp(cropLeft, 1.0);
    final cropBottom = crop.bottom.clamp(cropTop, 1.0);
    final normalizedRotation = rotationDegrees % 360;

    for (var y = 0; y < height; y++) {
      final targetY = cropTop + (y / height) * (cropBottom - cropTop);
      for (var x = 0; x < width; x++) {
        final targetX = cropLeft + (x / width) * (cropRight - cropLeft);
        final source = _rotatedToSource(
          targetX,
          targetY,
          normalizedRotation,
        );
        final sourceX = (source.$1 * (image.width - 1))
            .round()
            .clamp(0, image.width - 1);
        final sourceY = (source.$2 * (image.height - 1))
            .round()
            .clamp(0, image.height - 1);
        final rgb = _pixelRgb(image, sourceX, sourceY);

        input[inputIndex++] = normalize ? rgb.$1 / 255.0 : rgb.$1;
        input[inputIndex++] = normalize ? rgb.$2 / 255.0 : rgb.$2;
        input[inputIndex++] = normalize ? rgb.$3 / 255.0 : rgb.$3;
      }
    }

    return input;
  }

  static (double x, double y) _rotatedToSource(
    double x,
    double y,
    int rotationDegrees,
  ) {
    return switch (rotationDegrees) {
      90 => (y, 1 - x),
      180 => (1 - x, 1 - y),
      270 => (1 - y, x),
      _ => (x, y),
    };
  }

  static (int r, int g, int b) _pixelRgb(CameraImage image, int x, int y) {
    if (image.format.group == ImageFormatGroup.bgra8888) {
      final plane = image.planes.first;
      final index = y * plane.bytesPerRow + x * 4;
      final bytes = plane.bytes;
      return (bytes[index + 2], bytes[index + 1], bytes[index]);
    }

    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    final yIndex = y * yPlane.bytesPerRow + x;
    final uvX = (x / 2).floor();
    final uvY = (y / 2).floor();
    final uvIndex =
        uvY * uPlane.bytesPerRow + uvX * (uPlane.bytesPerPixel ?? 1);

    final yValue = yPlane.bytes[yIndex];
    final uValue = uPlane.bytes[uvIndex];
    final vValue = vPlane.bytes[uvIndex];

    final r = (yValue + 1.402 * (vValue - 128)).round();
    final g = (yValue - 0.344136 * (uValue - 128) - 0.714136 * (vValue - 128))
        .round();
    final b = (yValue + 1.772 * (uValue - 128)).round();

    return (_clampColor(r), _clampColor(g), _clampColor(b));
  }

  static int _clampColor(int value) => math.max(0, math.min(255, value));
}
