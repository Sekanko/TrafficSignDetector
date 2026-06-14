import 'dart:math' as math;
import 'dart:ui';

import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/available_model_preview.dart';
import '../models/detected_sign_preview.dart';

class DetectionCandidate {
  const DetectionCandidate({
    required this.boundingBox,
    required this.confidence,
  });

  final Rect boundingBox;
  final double confidence;
}

class DetectionOutputParser {
  const DetectionOutputParser._();

  static List<DetectedSignPreview> parse(
    Map<int, Object> outputs,
    AvailableModelPreview model,
  ) {
    if (outputs.length >= 4) {
      return _parseSsdOutputs(outputs, model);
    }

    return const [];
  }

  static List<DetectionCandidate> parseDetectorCandidates(
    Map<int, Object> outputs,
    double confidenceThreshold,
  ) {
    if (outputs.length >= 4) {
      return _parseSsdCandidates(outputs, confidenceThreshold);
    }

    return _parseYoloCandidates(outputs[0], confidenceThreshold);
  }

  static (int classIndex, double confidence)? parseClassification(
    Object? output,
  ) {
    if (output is! List) {
      return null;
    }

    final values = output.flatten<num>();
    if (values.isEmpty) {
      return null;
    }

    var bestIndex = 0;
    var bestValue = values.first.toDouble();
    for (var index = 1; index < values.length; index++) {
      final value = values[index].toDouble();
      if (value > bestValue) {
        bestIndex = index;
        bestValue = value;
      }
    }

    final confidence = _looksLikeProbability(values)
        ? bestValue
        : _softmaxConfidence(values, bestIndex);
    return (bestIndex, confidence);
  }

  static List<DetectionCandidate> _parseSsdCandidates(
    Map<int, Object> outputs,
    double confidenceThreshold,
  ) {
    final boxes = (outputs[0] as List).flatten<num>();
    final scores = (outputs[2] as List).flatten<num>();
    final rawCount = (outputs[3] as List).flatten<num>();
    final count = rawCount.isEmpty ? scores.length : rawCount.first.toInt();
    final detections = <DetectionCandidate>[];

    for (var index = 0; index < math.min(count, scores.length); index++) {
      final score = scores[index].toDouble();
      if (score < confidenceThreshold) {
        continue;
      }

      final boxIndex = index * 4;
      if (boxIndex + 3 >= boxes.length) {
        break;
      }

      detections.add(
        DetectionCandidate(
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

    return _nonMaxSuppression(detections);
  }

  static List<DetectionCandidate> _parseYoloCandidates(
    Object? output,
    double confidenceThreshold,
  ) {
    if (output is! List) {
      return const [];
    }

    final shape = output.shape;
    final values = output.flatten<num>();
    if (shape.length < 3 || values.isEmpty) {
      return const [];
    }

    final dimA = shape[shape.length - 2];
    final dimB = shape.last;
    final candidates = dimA <= dimB
        ? _parseYoloChannelsFirst(values, dimA, dimB, confidenceThreshold)
        : _parseYoloRows(values, dimB, confidenceThreshold);

    return _nonMaxSuppression(candidates);
  }

  static List<DetectionCandidate> _parseYoloChannelsFirst(
    List<num> values,
    int attributes,
    int count,
    double confidenceThreshold,
  ) {
    if (attributes < 5) {
      return const [];
    }

    final detections = <DetectionCandidate>[];
    for (var index = 0; index < count; index++) {
      final x = values[index].toDouble();
      final y = values[count + index].toDouble();
      final w = values[count * 2 + index].toDouble();
      final h = values[count * 3 + index].toDouble();
      final confidence = _yoloConfidence(values, attributes, count, index);

      if (confidence < confidenceThreshold) {
        continue;
      }

      detections.add(
        DetectionCandidate(
          confidence: confidence,
          boundingBox: _yoloRect(x: x, y: y, w: w, h: h),
        ),
      );
    }

    return detections;
  }

  static List<DetectionCandidate> _parseYoloRows(
    List<num> values,
    int attributes,
    double confidenceThreshold,
  ) {
    if (attributes < 5) {
      return const [];
    }

    final detections = <DetectionCandidate>[];
    for (
      var offset = 0;
      offset + attributes <= values.length;
      offset += attributes
    ) {
      final x = values[offset].toDouble();
      final y = values[offset + 1].toDouble();
      final w = values[offset + 2].toDouble();
      final h = values[offset + 3].toDouble();
      final confidence = _rowConfidence(values, offset, attributes);

      if (confidence < confidenceThreshold) {
        continue;
      }

      detections.add(
        DetectionCandidate(
          confidence: confidence,
          boundingBox: _yoloRect(x: x, y: y, w: w, h: h),
        ),
      );
    }

    return detections;
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

  static Rect _yoloRect({
    required double x,
    required double y,
    required double w,
    required double h,
  }) {
    final scale = math.max(
      math.max(x.abs(), y.abs()),
      math.max(w.abs(), h.abs()),
    );
    final normalizedX = scale > 2 ? x / 320.0 : x;
    final normalizedY = scale > 2 ? y / 320.0 : y;
    final normalizedW = scale > 2 ? w / 320.0 : w;
    final normalizedH = scale > 2 ? h / 320.0 : h;

    return _normalizedRect(
      left: normalizedX - normalizedW / 2,
      top: normalizedY - normalizedH / 2,
      right: normalizedX + normalizedW / 2,
      bottom: normalizedY + normalizedH / 2,
    );
  }

  static double _yoloConfidence(
    List<num> values,
    int attributes,
    int count,
    int index,
  ) {
    if (attributes == 5) {
      return values[count * 4 + index].toDouble();
    }

    final objectness = values[count * 4 + index].toDouble();
    var bestClassScore = 0.0;
    for (var attribute = 5; attribute < attributes; attribute++) {
      bestClassScore = math.max(
        bestClassScore,
        values[count * attribute + index].toDouble(),
      );
    }

    return objectness * bestClassScore;
  }

  static double _rowConfidence(List<num> values, int offset, int attributes) {
    if (attributes == 5) {
      return values[offset + 4].toDouble();
    }

    final objectness = values[offset + 4].toDouble();
    var bestClassScore = 0.0;
    for (var attribute = 5; attribute < attributes; attribute++) {
      bestClassScore = math.max(
        bestClassScore,
        values[offset + attribute].toDouble(),
      );
    }

    return objectness * bestClassScore;
  }

  static List<DetectionCandidate> _nonMaxSuppression(
    List<DetectionCandidate> candidates, {
    double iouThreshold = 0.45,
    int maxDetections = 20,
  }) {
    final sorted = [...candidates]
      ..sort((a, b) => b.confidence.compareTo(a.confidence));
    final kept = <DetectionCandidate>[];

    for (final candidate in sorted) {
      if (kept.length >= maxDetections) {
        break;
      }
      if (kept.any(
        (existing) =>
            _iou(existing.boundingBox, candidate.boundingBox) > iouThreshold,
      )) {
        continue;
      }
      kept.add(candidate);
    }

    return kept;
  }

  static double _iou(Rect a, Rect b) {
    final intersection = a.intersect(b);
    if (intersection.isEmpty) {
      return 0;
    }

    final intersectionArea = intersection.width * intersection.height;
    final unionArea =
        a.width * a.height + b.width * b.height - intersectionArea;
    if (unionArea <= 0) {
      return 0;
    }

    return intersectionArea / unionArea;
  }

  static bool _looksLikeProbability(List<num> values) {
    final sum = values.fold<double>(
      0,
      (total, value) => total + value.toDouble(),
    );
    return values.every((value) => value >= 0 && value <= 1) &&
        sum > 0.8 &&
        sum < 1.2;
  }

  static double _softmaxConfidence(List<num> values, int bestIndex) {
    final maxValue = values
        .map((value) => value.toDouble())
        .reduce((a, b) => math.max(a, b));
    final denominator = values.fold<double>(
      0,
      (total, value) => total + math.exp(value.toDouble() - maxValue),
    );
    if (denominator == 0) {
      return 0;
    }
    return math.exp(values[bestIndex].toDouble() - maxValue) / denominator;
  }

  static String _labelForClass(AvailableModelPreview model, int classIndex) {
    if (classIndex >= 0 && classIndex < model.labels.length) {
      return model.labels[classIndex];
    }
    return 'Znak ${classIndex + 1}';
  }
}
