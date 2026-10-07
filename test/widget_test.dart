// This is a basic Flutter widget test for YoloKhoApp.

import 'package:flutter_test/flutter_test.dart';
import 'package:yolo_kho/main.dart';

void main() {
  testWidgets('YoloKhoApp smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const YoloKhoApp());

    // Verify that app title exists.
    expect(find.text('Đếm Chai Lọ Kho (YOLOv8n)'), findsOneWidget);
  });
}
