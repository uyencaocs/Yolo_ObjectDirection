import 'package:flutter/material.dart';
import 'yolo_detector.dart';

class YoloBoxPainter extends CustomPainter {
  final List<Recognition> recognitions;
  final Size imageSize;
  final bool showLabels;

  YoloBoxPainter({
    required this.recognitions,
    required this.imageSize,
    this.showLabels = true,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (imageSize.width <= 0 || imageSize.height <= 0) return;

    // Calculate scaling ratios
    final double scaleX = size.width;
    final double scaleY = size.height;

    // Distinct vibrant color palette for different detection classes
    final List<Color> colors = [
      const Color(0xFF00E676), // Vibrant Green
      const Color(0xFF2979FF), // Bright Blue
      const Color(0xFFFF9100), // Orange
      const Color(0xFFD500F9), // Purple
      const Color(0xFF00E5FF), // Cyan
      const Color(0xFFFFEA00), // Yellow
    ];

    for (int i = 0; i < recognitions.length; i++) {
      final recognition = recognitions[i];
      final color = colors[recognition.classId % colors.length];

      final rect = Rect.fromLTRB(
        recognition.location.left * scaleX,
        recognition.location.top * scaleY,
        recognition.location.right * scaleX,
        recognition.location.bottom * scaleY,
      );

      // 1. Draw bounding box rectangle
      final strokePaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3.0;

      final fillPaint = Paint()
        ..color = color.withAlpha(45)
        ..style = PaintingStyle.fill;

      canvas.drawRRect(
        RRect.fromRectAndRadius(rect, const Radius.circular(6)),
        fillPaint,
      );
      canvas.drawRRect(
        RRect.fromRectAndRadius(rect, const Radius.circular(6)),
        strokePaint,
      );

      // 2. Draw Number Circle Badge on Top Left
      const double badgeRadius = 13.0;
      final Offset badgeCenter = Offset(rect.left + 15, rect.top + 15);

      final badgeBgPaint = Paint()
        ..color = color
        ..style = PaintingStyle.fill;

      canvas.drawCircle(badgeCenter, badgeRadius, badgeBgPaint);

      final numTextPainter = TextPainter(
        text: TextSpan(
          text: '${i + 1}',
          style: const TextStyle(
            color: Colors.white,
            fontSize: 12,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      numTextPainter.layout();
      numTextPainter.paint(
        canvas,
        Offset(
          badgeCenter.dx - numTextPainter.width / 2,
          badgeCenter.dy - numTextPainter.height / 2,
        ),
      );

      // 3. Draw Label & Confidence Tag
      if (showLabels) {
        final labelText = '${recognition.label} ${(recognition.score * 100).toStringAsFixed(0)}%';
        final textSpan = TextSpan(
          text: labelText,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 11,
            fontWeight: FontWeight.w600,
          ),
        );

        final textPainter = TextPainter(
          text: textSpan,
          textDirection: TextDirection.ltr,
        );
        textPainter.layout();

        const double textPaddingHorizontal = 6.0;
        const double textPaddingVertical = 4.0;

        final double tagLeft = rect.left + 32;
        final double tagTop = (rect.top - 20 < 0) ? rect.top + 4 : rect.top - 22;

        final tagRect = Rect.fromLTWH(
          tagLeft,
          tagTop,
          textPainter.width + textPaddingHorizontal * 2,
          textPainter.height + textPaddingVertical * 2,
        );

        final tagBgPaint = Paint()
          ..color = Colors.black87
          ..style = PaintingStyle.fill;

        canvas.drawRRect(
          RRect.fromRectAndRadius(tagRect, const Radius.circular(4)),
          tagBgPaint,
        );

        textPainter.paint(
          canvas,
          Offset(
            tagLeft + textPaddingHorizontal,
            tagTop + textPaddingVertical,
          ),
        );
      }
    }
  }

  @override
  bool shouldRepaint(covariant YoloBoxPainter oldDelegate) {
    return oldDelegate.recognitions != recognitions ||
        oldDelegate.imageSize != imageSize ||
        oldDelegate.showLabels != showLabels;
  }
}
