import 'package:flutter/material.dart';

import '../models/detected_sign_preview.dart';

class RecentSignsStrip extends StatelessWidget {
  const RecentSignsStrip({
    required this.signs,
    super.key,
  });

  final List<DetectedSignPreview> signs;

  @override
  Widget build(BuildContext context) {
    final recentSigns = signs.take(3).toList();

    return Container(
      height: 88,
      width: double.infinity,
      margin: const EdgeInsets.fromLTRB(12, 0, 12, 12),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.66),
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: Colors.white.withValues(alpha: 0.14)),
      ),
      child: recentSigns.isEmpty
          ? const Center(
              child: Text(
                'Nie wykryto żadnych znaków :(',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Colors.white70,
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
            )
          : Row(
              children: recentSigns
                  .map(
                    (sign) => Padding(
                      padding: const EdgeInsets.only(right: 10),
                      child: _DetectedSignCutout(sign: sign),
                    ),
                  )
                  .toList(),
            ),
    );
  }
}

class _DetectedSignCutout extends StatelessWidget {
  const _DetectedSignCutout({required this.sign});

  final DetectedSignPreview sign;

  @override
  Widget build(BuildContext context) {
    return AspectRatio(
      aspectRatio: 1,
      child: Container(
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.92),
          borderRadius: BorderRadius.circular(16),
        ),
        clipBehavior: Clip.antiAlias,
        child: sign.thumbnail == null
            ? Center(
                child: Text(
                  sign.label,
                  textAlign: TextAlign.center,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    color: Colors.black87,
                    fontSize: 13,
                    fontWeight: FontWeight.w900,
                  ),
                ),
              )
            : Image(
                image: sign.thumbnail!,
                fit: BoxFit.cover,
              ),
      ),
    );
  }
}
