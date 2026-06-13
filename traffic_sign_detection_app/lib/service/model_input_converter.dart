import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class ModelInputConverter {
  const ModelInputConverter._();

  static Object? cameraImageToInput({
    required CameraImage image,
    required List<int> inputShape,
    required int fallbackSize,
    required TensorType type,
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
  }) {
    final input = List<num>.filled(width * height * 3, 0);
    var inputIndex = 0;

    for (var y = 0; y < height; y++) {
      final sourceY = (y * image.height / height).floor();
      for (var x = 0; x < width; x++) {
        final sourceX = (x * image.width / width).floor();
        final rgb = _pixelRgb(image, sourceX, sourceY);

        input[inputIndex++] = normalize ? rgb.$1 / 255.0 : rgb.$1;
        input[inputIndex++] = normalize ? rgb.$2 / 255.0 : rgb.$2;
        input[inputIndex++] = normalize ? rgb.$3 / 255.0 : rgb.$3;
      }
    }

    return input;
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
