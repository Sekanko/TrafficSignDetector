import 'package:flutter/material.dart';

import '../models/detected_sign_preview.dart';

class DetectionOverlay extends StatelessWidget {
  const DetectionOverlay({required this.detections, super.key});

  final List<DetectedSignPreview> detections;

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: CustomPaint(
        painter: _DetectionOverlayPainter(detections),
        size: Size.infinite,
      ),
    );
  }
}

class _DetectionOverlayPainter extends CustomPainter {
  const _DetectionOverlayPainter(this.detections);

  final List<DetectedSignPreview> detections;

  @override
  void paint(Canvas canvas, Size size) {
    final borderPaint = Paint()
      ..color = const Color(0xFF28E07B)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;
    final fillPaint = Paint()
      ..color = Colors.black.withValues(alpha: 0.72)
      ..style = PaintingStyle.fill;

    for (final detection in detections) {
      final box = detection.boundingBox;
      if (box == null || box.isEmpty) {
        continue;
      }

      final rect = Rect.fromLTRB(
        box.left * size.width,
        box.top * size.height,
        box.right * size.width,
        box.bottom * size.height,
      );

      canvas.drawRRect(
        RRect.fromRectAndRadius(rect, const Radius.circular(10)),
        borderPaint,
      );

      final confidence = detection.confidence == null
          ? ''
          : ' ${(detection.confidence! * 100).round()}%';
      final textPainter = TextPainter(
        text: TextSpan(
          text: '${detection.label}$confidence',
          style: const TextStyle(
            color: Colors.white,
            fontSize: 13,
            fontWeight: FontWeight.w800,
          ),
        ),
        maxLines: 1,
        textDirection: TextDirection.ltr,
      )..layout(maxWidth: size.width * 0.8);

      final labelRect = Rect.fromLTWH(
        rect.left,
        (rect.top - textPainter.height - 8).clamp(0, size.height),
        textPainter.width + 16,
        textPainter.height + 8,
      );

      canvas.drawRRect(
        RRect.fromRectAndRadius(labelRect, const Radius.circular(8)),
        fillPaint,
      );
      textPainter.paint(canvas, Offset(labelRect.left + 8, labelRect.top + 4));
    }
  }

  @override
  bool shouldRepaint(covariant _DetectionOverlayPainter oldDelegate) {
    return oldDelegate.detections != detections;
  }
}
