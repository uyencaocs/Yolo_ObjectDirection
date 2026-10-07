import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class Recognition {
  final int id;
  final String label;
  final double score;
  final Rect location; // In normalized 0.0 - 1.0 coordinates relative to original image
  final int classId;

  Recognition({
    required this.id,
    required this.label,
    required this.score,
    required this.location,
    required this.classId,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'label': label,
        'score': '${(score * 100).toStringAsFixed(1)}%',
        'classId': classId,
      };
}

class LetterboxImage {
  final img.Image image;
  final double scale;
  final int padX;
  final int padY;
  final int origWidth;
  final int origHeight;

  LetterboxImage({
    required this.image,
    required this.scale,
    required this.padX,
    required this.padY,
    required this.origWidth,
    required this.origHeight,
  });
}

class YoloDetector {
  Interpreter? _interpreter;
  bool _isLoaded = false;

  static const int inputSize = 640;

  // COCO class indices related to bottles, containers, glasses, cups & vases:
  // 39: Chai/Lọ, 40: Ly rượu, 41: Cốc/Ly, 45: Tách/Bát, 75: Bình/Lọ hoa
  static const Set<int> bottleClassIds = {39, 40, 41, 45, 75};

  // Từ điển Tiếng Việt chuẩn cho 80 lớp COCO
  static const Map<int, String> vietnameseLabels = {
    0: 'Người',
    1: 'Xe đạp',
    2: 'Ô tô',
    3: 'Xe máy',
    4: 'Máy bay',
    5: 'Xe buýt',
    6: 'Tàu hỏa',
    7: 'Xe tải',
    8: 'Thuyền',
    9: 'Đèn giao thông',
    10: 'Trụ chữa cháy',
    11: 'Biển dừng',
    12: 'Đồng hồ đỗ xe',
    13: 'Ghế dài',
    14: 'Chim',
    15: 'Mèo',
    16: 'Chó',
    17: 'Ngựa',
    18: 'Cừu',
    19: 'Bò',
    20: 'Voi',
    21: 'Gấu',
    22: 'Ngựa vằn',
    23: 'Hươu cao cổ',
    24: 'Balo',
    25: 'Ô / Dù',
    26: 'Túi xách',
    27: 'Cà vạt',
    28: 'Vali',
    29: 'Đĩa bay cản',
    30: 'Ván trượt tuyết',
    31: 'Ván trượt',
    32: 'Bóng thể thao',
    33: 'Diều',
    34: 'Gậy bóng chày',
    35: 'Găng tay bóng chày',
    36: 'Ván trượt',
    37: 'Ván lướt sóng',
    38: 'Vợt tennis',
    39: 'Chai / Lọ',
    40: 'Ly rượu',
    41: 'Cốc / Ly',
    42: 'Nĩa',
    43: 'Dao',
    44: 'Muỗng',
    45: 'Tách / Bát',
    46: 'Chuối',
    47: 'Táo',
    48: 'Bánh mì kẹp',
    49: 'Cam',
    50: 'Bông cải xanh',
    51: 'Cà rốt',
    52: 'Xúc xích',
    53: 'Bánh pizza',
    54: 'Bánh donut',
    55: 'Bánh ngọt',
    56: 'Ghế',
    57: 'Ghế sofa',
    58: 'Chậu cây',
    59: 'Giường',
    60: 'Bàn ăn',
    61: 'Bồn cầu',
    62: 'Tivi',
    63: 'Máy tính xách tay',
    64: 'Chuột máy tính',
    65: 'Điều khiển',
    66: 'Bàn phím',
    67: 'Điện thoại',
    68: 'Lò vi sóng',
    69: 'Lò nướng',
    70: 'Máy nướng bánh mì',
    71: 'Bồn rửa',
    72: 'Tủ lạnh',
    73: 'Sách',
    74: 'Đồng hồ',
    75: 'Bình / Lọ hoa',
    76: 'Kéo',
    77: 'Gấu bông',
    78: 'Máy sấy tóc',
    79: 'Bàn chải đánh răng',
  };

  bool get isLoaded => _isLoaded;

  Future<void> loadModel() async {
    try {
      final options = InterpreterOptions();
      if (!kIsWeb && (Platform.isAndroid || Platform.isIOS)) {
        options.threads = 4;
      }

      _interpreter = await Interpreter.fromAsset(
        'assets/yolov8n.tflite',
        options: options,
      );
      _isLoaded = true;

      final inputTensor = _interpreter!.getInputTensor(0);
      final outputTensor = _interpreter!.getOutputTensor(0);

      debugPrint("YOLOv8 Float32 Model loaded successfully.");
      debugPrint("Input Tensor: shape=${inputTensor.shape}, type=${inputTensor.type}");
      debugPrint("Output Tensor: shape=${outputTensor.shape}, type=${outputTensor.type}");
    } catch (e) {
      debugPrint("Error loading YOLO model: $e");
      _isLoaded = false;
    }
  }

  Future<List<Recognition>> detectObjects(
    Uint8List imageBytes, {
    double confidenceThreshold = 0.20,
    double iouThreshold = 0.40,
    Set<int>? targetClassIds,
  }) async {
    final originalImage = img.decodeImage(imageBytes);
    if (originalImage == null) return [];

    if (!_isLoaded || _interpreter == null) {
      return [];
    }

    try {
      // 1. Preprocess with Letterboxing
      final letterbox = _createLetterboxImage(originalImage, inputSize);

      final inputTensor = _interpreter!.getInputTensor(0);
      final inputShape = inputTensor.shape; // e.g. [1, 640, 640, 3] or [1, 3, 640, 640]
      final isChannelFirst = inputShape.length == 4 && inputShape[1] == 3;

      dynamic inputData;

      if (isChannelFirst) {
        // [1, 3, 640, 640]
        inputData = List.generate(
          1,
          (_) => List.generate(
            3,
            (c) => List.generate(
              inputSize,
              (y) => List.generate(
                inputSize,
                (x) {
                  final pixel = letterbox.image.getPixel(x, y);
                  if (c == 0) return pixel.r / 255.0;
                  if (c == 1) return pixel.g / 255.0;
                  return pixel.b / 255.0;
                },
              ),
            ),
          ),
        );
      } else {
        // [1, 640, 640, 3]
        inputData = List.generate(
          1,
          (_) => List.generate(
            inputSize,
            (y) => List.generate(
              inputSize,
              (x) {
                final pixel = letterbox.image.getPixel(x, y);
                return [
                  pixel.r / 255.0,
                  pixel.g / 255.0,
                  pixel.b / 255.0,
                ];
              },
            ),
          ),
        );
      }

      final outputTensor = _interpreter!.getOutputTensor(0);
      final outputShape = outputTensor.shape; // e.g. [1, 84, 8400] or [1, 8400, 84]

      dynamic outputData;
      if (outputShape.length == 3) {
        outputData = List.generate(
          outputShape[0],
          (_) => List.generate(
            outputShape[1],
            (_) => List<double>.filled(outputShape[2], 0.0),
          ),
        );
      } else {
        outputData = List.filled(outputShape.reduce((a, b) => a * b), 0.0);
      }

      // Run inference directly
      _interpreter!.run(inputData, outputData);

      // Parse detections
      List<Recognition> rawDetections = _parseYolov8Output(
        outputData,
        outputShape,
        letterbox: letterbox,
        confidenceThreshold: confidenceThreshold,
        targetClassIds: targetClassIds,
      );

      debugPrint("Raw detections count: ${rawDetections.length}");

      // Apply Non-Maximum Suppression (NMS)
      List<Recognition> nmsResults = _applyNMS(
        rawDetections,
        iouThreshold: iouThreshold,
      );

      debugPrint("NMS results count: ${nmsResults.length}");
      return nmsResults;
    } catch (e, stack) {
      debugPrint("Inference error: $e\n$stack");
      return [];
    }
  }

  LetterboxImage _createLetterboxImage(img.Image src, int targetSize) {
    final origW = src.width;
    final origH = src.height;

    final double scale = min(targetSize / origW, targetSize / origH);
    final newW = (origW * scale).round();
    final newH = (origH * scale).round();

    final padX = (targetSize - newW) ~/ 2;
    final padY = (targetSize - newH) ~/ 2;

    final resized = img.copyResize(src, width: newW, height: newH);

    final canvas = img.Image(
      width: targetSize,
      height: targetSize,
      numChannels: 3,
    );
    canvas.clear(img.ColorRgb8(114, 114, 114));

    img.compositeImage(
      canvas,
      resized,
      dstX: padX,
      dstY: padY,
    );

    return LetterboxImage(
      image: canvas,
      scale: scale,
      padX: padX,
      padY: padY,
      origWidth: origW,
      origHeight: origH,
    );
  }

  double _sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
  }

  List<Recognition> _parseYolov8Output(
    dynamic outputData,
    List<int> shape, {
    required LetterboxImage letterbox,
    required double confidenceThreshold,
    Set<int>? targetClassIds,
  }) {
    List<Recognition> detections = [];
    if (shape.length < 2) return detections;

    bool is3D = shape.length >= 3;
    int numRows = is3D ? shape[1] : shape[0];
    int numCols = is3D ? shape[2] : shape[1];

    bool isChannelsFirst = numRows < numCols; // [1, 84, 8400] vs [1, 8400, 84]

    int numAnchors = isChannelsFirst ? numCols : numRows;
    int numAttributes = isChannelsFirst ? numRows : numCols;
    int numClasses = numAttributes - 4;
    if (numClasses <= 0) return detections;

    double getValue(int row, int col) {
      try {
        if (is3D) {
          if (outputData is List && outputData.isNotEmpty && outputData[0] is List) {
            if (isChannelsFirst) {
              return (outputData[0][row][col] as num).toDouble();
            } else {
              return (outputData[0][col][row] as num).toDouble();
            }
          }
        } else {
          if (outputData is List) {
            if (outputData.isNotEmpty && outputData[0] is List) {
              return isChannelsFirst
                  ? (outputData[row][col] as num).toDouble()
                  : (outputData[col][row] as num).toDouble();
            } else {
              int index = isChannelsFirst ? (row * numAnchors + col) : (col * numAttributes + row);
              if (index >= 0 && index < outputData.length) {
                return (outputData[index] as num).toDouble();
              }
            }
          }
        }
      } catch (_) {}
      return 0.0;
    }

    double globalMaxScore = -999.0;
    int globalBestClass = -1;

    int detectionId = 0;

    for (int i = 0; i < numAnchors; i++) {
      double cx = getValue(0, i);
      double cy = getValue(1, i);
      double w = getValue(2, i);
      double h = getValue(3, i);

      // Find max class confidence score
      double maxScore = -999.0;
      int bestClassId = -1;

      for (int c = 0; c < numClasses; c++) {
        double score = getValue(4 + c, i);
        if (score > maxScore) {
          maxScore = score;
          bestClassId = c;
        }
      }

      if (maxScore > globalMaxScore) {
        globalMaxScore = maxScore;
        globalBestClass = bestClassId;
      }

      // If maxScore is outside [0.0 - 1.0], apply Sigmoid conversion
      double prob = maxScore;
      if (maxScore < 0.0 || maxScore > 1.0) {
        prob = _sigmoid(maxScore);
      }

      if (prob >= confidenceThreshold && bestClassId >= 0) {
        if (targetClassIds != null && !targetClassIds.contains(bestClassId)) {
          continue;
        }

        // YOLOv8 box coordinates: if <= 1.0, scale up by inputSize (640)
        double cxPx = (cx <= 1.0) ? cx * inputSize : cx;
        double cyPx = (cy <= 1.0) ? cy * inputSize : cy;
        double wPx = (w <= 1.0) ? w * inputSize : w;
        double hPx = (h <= 1.0) ? h * inputSize : h;

        // Box in letterbox 640x640 space (pixels)
        double boxLeftPx = cxPx - wPx / 2;
        double boxTopPx = cyPx - hPx / 2;
        double boxRightPx = cxPx + wPx / 2;
        double boxBottomPx = cyPx + hPx / 2;

        // Remove letterbox padding and scale back to original image dimensions
        double origLeftPx = (boxLeftPx - letterbox.padX) / letterbox.scale;
        double origTopPx = (boxTopPx - letterbox.padY) / letterbox.scale;
        double origRightPx = (boxRightPx - letterbox.padX) / letterbox.scale;
        double origBottomPx = (boxBottomPx - letterbox.padY) / letterbox.scale;

        // Convert to normalized 0.0 - 1.0 coordinates relative to original image
        double left = (origLeftPx / letterbox.origWidth).clamp(0.0, 1.0);
        double top = (origTopPx / letterbox.origHeight).clamp(0.0, 1.0);
        double right = (origRightPx / letterbox.origWidth).clamp(0.0, 1.0);
        double bottom = (origBottomPx / letterbox.origHeight).clamp(0.0, 1.0);

        double width = (right - left).clamp(0.0, 1.0 - left);
        double height = (bottom - top).clamp(0.0, 1.0 - top);

        // Filter out extreme tiny noise boxes
        if (width <= 0.003 || height <= 0.003) continue;

        String labelVietnamese = vietnameseLabels[bestClassId] ?? 'Vật thể $bestClassId';

        detections.add(
          Recognition(
            id: detectionId++,
            label: labelVietnamese,
            score: prob,
            location: Rect.fromLTWH(left, top, width, height),
            classId: bestClassId,
          ),
        );
      }
    }

    debugPrint("Global max score: $globalMaxScore (prob: ${globalMaxScore < 0.0 || globalMaxScore > 1.0 ? _sigmoid(globalMaxScore) : globalMaxScore}), Best class: $globalBestClass (${vietnameseLabels[globalBestClass]})");

    return detections;
  }

  List<Recognition> _applyNMS(
    List<Recognition> boxes, {
    required double iouThreshold,
  }) {
    boxes.sort((a, b) => b.score.compareTo(a.score));

    List<Recognition> selected = [];
    List<bool> active = List.filled(boxes.length, true);

    for (int i = 0; i < boxes.length; i++) {
      if (!active[i]) continue;

      final boxA = boxes[i];
      selected.add(boxA);

      for (int j = i + 1; j < boxes.length; j++) {
        if (!active[j]) continue;

        final boxB = boxes[j];
        if (_calculateIoU(boxA.location, boxB.location) >= iouThreshold) {
          active[j] = false;
        }
      }
    }

    return selected;
  }

  double _calculateIoU(Rect a, Rect b) {
    final intersection = a.intersect(b);
    if (intersection.width <= 0 || intersection.height <= 0) return 0.0;

    final intersectionArea = intersection.width * intersection.height;
    final areaA = a.width * a.height;
    final areaB = b.width * b.height;

    final unionArea = areaA + areaB - intersectionArea;
    if (unionArea <= 0) return 0.0;

    return intersectionArea / unionArea;
  }

  void dispose() {
    _interpreter?.close();
  }
}
