import 'package:flutter/widgets.dart';

class DetectedSignPreview {
  const DetectedSignPreview({
    required this.label,
    this.thumbnail,
  });

  final String label;
  final ImageProvider? thumbnail;
}
