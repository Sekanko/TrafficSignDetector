import 'dart:math' as math;
import 'dart:ui';

import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/available_model_preview.dart';
import '../models/detected_sign_preview.dart';

class DetectionOutputParser {
  const DetectionOutputParser._();

  static List<DetectedSignPreview> parse(
    Map<int, Object> outputs,
    AvailableModelPreview model,
  ) {
    if (outputs.length >= 4) {
      return _parseSsdOutputs(outputs, model);
    }

    return _parseSingleDetectionOutput(outputs[0], model);
  }

  static List<DetectedSignPreview> _parseSsdOutputs(
    Map<int, Object> outputs,
    AvailableModelPreview model,
  ) {
    final boxes = (outputs[0] as List).flatten<num>();
    final classes = (outputs[1] as List).flatten<num>();
    final scores = (outputs[2] as List).flatten<num>();
    final rawCount = (outputs[3] as List).flatten<num>();
    final count = rawCount.isEmpty ? scores.length : rawCount.first.toInt();
    final detections = <DetectedSignPreview>[];

    for (var index = 0; index < math.min(count, scores.length); index++) {
      final score = scores[index].toDouble();
      if (score < model.confidenceThreshold) {
        continue;
      }

      final boxIndex = index * 4;
      if (boxIndex + 3 >= boxes.length) {
        break;
      }

      final classIndex = classes[index].round();
      detections.add(
        DetectedSignPreview(
          label: _labelForClass(model, classIndex),
          confidence: score,
          boundingBox: _normalizedRect(
            left: boxes[boxIndex + 1].toDouble(),
            top: boxes[boxIndex].toDouble(),
            right: boxes[boxIndex + 3].toDouble(),
            bottom: boxes[boxIndex + 2].toDouble(),
          ),
        ),
      );
    }

    return detections;
  }

  static List<DetectedSignPreview> _parseSingleDetectionOutput(
    Object? output,
    AvailableModelPreview model,
  ) {
    if (output is! List) {
      return const [];
    }

    final shape = output.shape;
    final values = output.flatten<num>();
    if (shape.length < 3 || values.isEmpty) {
      return const [];
    }

    final predictionWidth = shape.last;
    if (predictionWidth < 6) {
      return const [];
    }

    final detections = <DetectedSignPreview>[];
    for (
      var offset = 0;
      offset + predictionWidth <= values.length;
      offset += predictionWidth
    ) {
      final x = values[offset].toDouble();
      final y = values[offset + 1].toDouble();
      final w = values[offset + 2].toDouble();
      final h = values[offset + 3].toDouble();
      final objectness = values[offset + 4].toDouble();

      var bestClassIndex = 0;
      var bestClassScore = 0.0;
      for (var index = 5; index < predictionWidth; index++) {
        final score = values[offset + index].toDouble();
        if (score > bestClassScore) {
          bestClassScore = score;
          bestClassIndex = index - 5;
        }
      }

      final confidence = objectness * bestClassScore;
      if (confidence < model.confidenceThreshold) {
        continue;
      }

      detections.add(
        DetectedSignPreview(
          label: _labelForClass(model, bestClassIndex),
          confidence: confidence,
          boundingBox: _normalizedRect(
            left: x - w / 2,
            top: y - h / 2,
            right: x + w / 2,
            bottom: y + h / 2,
          ),
        ),
      );
    }

    return detections;
  }

  static Rect _normalizedRect({
    required double left,
    required double top,
    required double right,
    required double bottom,
  }) {
    return Rect.fromLTRB(
      left.clamp(0.0, 1.0),
      top.clamp(0.0, 1.0),
      right.clamp(0.0, 1.0),
      bottom.clamp(0.0, 1.0),
    );
  }

  static String _labelForClass(AvailableModelPreview model, int classIndex) {
    if (classIndex >= 0 && classIndex < model.labels.length) {
      return model.labels[classIndex];
    }
    return 'Znak ${classIndex + 1}';
  }
}
