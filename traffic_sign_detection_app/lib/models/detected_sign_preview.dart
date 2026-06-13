import 'package:flutter/widgets.dart';

class DetectedSignPreview {
  const DetectedSignPreview({
    required this.label,
    this.confidence,
    this.boundingBox,
    this.thumbnail,
  });

  final String label;
  final double? confidence;
  final Rect? boundingBox;
  final ImageProvider? thumbnail;
}
