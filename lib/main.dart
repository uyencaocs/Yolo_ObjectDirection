import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'yolo_detector.dart';
import 'box_painter.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const YoloKhoApp());
}

class YoloKhoApp extends StatelessWidget {
  const YoloKhoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Đếm Chai Lọ Kho Hàng',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF0F4C81), // Inventory Indigo Blue
          primary: const Color(0xFF0F4C81),
          secondary: const Color(0xFF00A86B), // Emerald Green
          brightness: Brightness.light,
        ),
        appBarTheme: const AppBarTheme(
          centerTitle: true,
          elevation: 0,
        ),
      ),
      home: const WarehouseCounterPage(),
    );
  }
}

class ScanRecord {
  final String id;
  final DateTime timestamp;
  final int count;
  final String filterType;
  final Uint8List imageBytes;

  ScanRecord({
    required this.id,
    required this.timestamp,
    required this.count,
    required this.filterType,
    required this.imageBytes,
  });
}

class WarehouseCounterPage extends StatefulWidget {
  const WarehouseCounterPage({super.key});

  @override
  State<WarehouseCounterPage> createState() => _WarehouseCounterPageState();
}

class _WarehouseCounterPageState extends State<WarehouseCounterPage> {
  final YoloDetector _detector = YoloDetector();
  final ImagePicker _picker = ImagePicker();

  Uint8List? _selectedImageBytes;
  Size? _imageSize;
  List<Recognition> _recognitions = [];
  bool _isLoading = false;
  bool _isModelReady = false;

  // Optimized internal parameters
  static const double _confidenceThreshold = 0.15; // Lower threshold to capture cups/bottles easily
  static const double _iouThreshold = 0.40;

  int _filterMode = 1; // Mặc định: Chai, Ly, Cốc & Lọ

  final List<ScanRecord> _scanHistory = [];

  @override
  void initState() {
    super.initState();
    _initDetector();
  }

  Future<void> _initDetector() async {
    setState(() => _isLoading = true);
    await _detector.loadModel();
    setState(() {
      _isModelReady = _detector.isLoaded;
      _isLoading = false;
    });
  }

  Set<int>? _getTargetClassIds() {
    switch (_filterMode) {
      case 0:
        return {39}; // Chỉ chai / lọ
      case 1:
        return YoloDetector.bottleClassIds; // Chai, ly, cốc, bát, lọ
      case 2:
      default:
        return null; // Tất cả vật thể
    }
  }

  String _getFilterModeName() {
    switch (_filterMode) {
      case 0:
        return 'Chỉ Chai / Lọ';
      case 1:
        return 'Chai, Ly, Cốc & Lọ';
      case 2:
      default:
        return 'Tất cả vật thể';
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: source,
        maxWidth: 1920,
        maxHeight: 1920,
      );

      if (pickedFile == null) return;

      final bytes = await pickedFile.readAsBytes();
      await _processImageBytes(bytes);
    } catch (e) {
      _showSnackBar('Lỗi chọn hình ảnh: $e');
    }
  }

  Future<void> _processImageBytes(Uint8List bytes) async {
    setState(() {
      _isLoading = true;
      _selectedImageBytes = bytes;
    });

    final decodedImage = await decodeImageFromList(bytes);
    _imageSize = Size(
      decodedImage.width.toDouble(),
      decodedImage.height.toDouble(),
    );

    final results = await _detector.detectObjects(
      bytes,
      confidenceThreshold: _confidenceThreshold,
      iouThreshold: _iouThreshold,
      targetClassIds: _getTargetClassIds(),
    );

    setState(() {
      _recognitions = results;
      _isLoading = false;
    });

    // Save to scan history
    _scanHistory.insert(
      0,
      ScanRecord(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        timestamp: DateTime.now(),
        count: results.length,
        filterType: _getFilterModeName(),
        imageBytes: bytes,
      ),
    );
  }

  void _reDetect() {
    if (_selectedImageBytes != null) {
      _processImageBytes(_selectedImageBytes!);
    }
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  @override
  void dispose() {
    _detector.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final primaryColor = Theme.of(context).colorScheme.primary;

    return Scaffold(
      appBar: AppBar(
        title: const Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.inventory_2_rounded, color: Colors.amber),
            SizedBox(width: 8),
            Text(
              'Đếm Chai Lọ Kho (YOLOv8n)',
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
            ),
          ],
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.history_rounded),
            tooltip: 'Lịch sử đếm',
            onPressed: () => _showHistorySheet(context),
          ),
        ],
      ),
      body: Column(
        children: [
          // 1. Status & Filter Header
          _buildControlHeader(),

          // 2. Main Image / Detection Display
          Expanded(
            child: _selectedImageBytes == null
                ? _buildEmptyPlaceholder()
                : _buildImageDisplayArea(),
          ),

          // 3. Count Results Banner & Bottom Action Panel
          _buildBottomActionPanel(primaryColor),
        ],
      ),
    );
  }

  Widget _buildControlHeader() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.grey.shade100,
        border: Border(bottom: BorderSide(color: Colors.grey.shade300)),
      ),
      child: Row(
        children: [
          // Status Badge
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: _isModelReady ? Colors.green.shade100 : Colors.orange.shade100,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                color: _isModelReady ? Colors.green : Colors.orange,
              ),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  _isModelReady ? Icons.check_circle : Icons.hourglass_top,
                  size: 16,
                  color: _isModelReady ? Colors.green.shade800 : Colors.orange.shade800,
                ),
                const SizedBox(width: 6),
                Text(
                  _isModelReady ? 'Hệ thống sẵn sàng' : 'Đang tải mô hình...',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                    color: _isModelReady ? Colors.green.shade900 : Colors.orange.shade900,
                  ),
                ),
              ],
            ),
          ),
          const Spacer(),

          // Class Filter Selector Dropdown
          DropdownButton<int>(
            value: _filterMode,
            underline: const SizedBox(),
            borderRadius: BorderRadius.circular(12),
            icon: const Icon(Icons.filter_alt_rounded, size: 20),
            items: const [
              DropdownMenuItem(
                value: 0,
                child: Text(' Chỉ Chai / Lọ', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
              ),
              DropdownMenuItem(
                value: 1,
                child: Text(' Chai, Ly, Cốc & Lọ', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
              ),
              DropdownMenuItem(
                value: 2,
                child: Text(' Tất cả vật thể', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
              ),
            ],
            onChanged: (val) {
              if (val != null && val != _filterMode) {
                setState(() => _filterMode = val);
                _reDetect();
              }
            },
          ),
        ],
      ),
    );
  }

  Widget _buildEmptyPlaceholder() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                color: Colors.blue.shade50,
                shape: BoxShape.circle,
              ),
              child: Icon(
                Icons.add_a_photo_rounded,
                size: 72,
                color: Theme.of(context).colorScheme.primary,
              ),
            ),
            const SizedBox(height: 20),
            const Text(
              'Chụp hoặc Chọn Hình Ảnh Chai Lọ Kho Hàng',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Hệ thống tự động nhận diện & đếm chính xác số lượng chai lọ bằng AI',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 13, color: Colors.grey.shade600),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildImageDisplayArea() {
    return Stack(
      children: [
        Center(
          child: InteractiveViewer(
            maxScale: 5.0,
            child: AspectRatio(
              aspectRatio: (_imageSize != null)
                  ? _imageSize!.width / _imageSize!.height
                  : 1.0,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  Image.memory(
                    _selectedImageBytes!,
                    fit: BoxFit.fill,
                  ),
                  if (_imageSize != null)
                    CustomPaint(
                      painter: YoloBoxPainter(
                        recognitions: _recognitions,
                        imageSize: _imageSize!,
                      ),
                    ),
                ],
              ),
            ),
          ),
        ),
        if (_isLoading)
          Container(
            color: Colors.black45,
            child: const Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  CircularProgressIndicator(color: Colors.white),
                  SizedBox(height: 12),
                  Text(
                    'Đang phân tích hình ảnh...',
                    style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                  ),
                ],
              ),
            ),
          ),
      ],
    );
  }

  Widget _buildBottomActionPanel(Color primaryColor) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withAlpha(20),
            blurRadius: 8,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Total Count Banner
          if (_selectedImageBytes != null) ...[
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              decoration: BoxDecoration(
                color: _recognitions.isNotEmpty ? const Color(0xFFE8F5E9) : const Color(0xFFFFF3E0),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                  color: _recognitions.isNotEmpty ? const Color(0xFF81C784) : const Color(0xFFFFB74D),
                ),
              ),
              child: Row(
                children: [
                  Icon(
                    _recognitions.isNotEmpty ? Icons.task_alt_rounded : Icons.info_outline_rounded,
                    color: _recognitions.isNotEmpty ? const Color(0xFF2E7D32) : const Color(0xFFE65100),
                    size: 28,
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'KẾT QUẢ ĐẾM TRONG KHO',
                          style: TextStyle(
                            fontSize: 11,
                            fontWeight: FontWeight.bold,
                            letterSpacing: 0.8,
                            color: Colors.grey.shade700,
                          ),
                        ),
                        Text(
                          'Phát hiện ${_recognitions.length} chai lọ / vật thể',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: _recognitions.isNotEmpty ? const Color(0xFF1B5E20) : const Color(0xFFBF360C),
                          ),
                        ),
                      ],
                    ),
                  ),
                  if (_recognitions.isNotEmpty)
                    TextButton.icon(
                      onPressed: () => _showDetailsSheet(context),
                      icon: const Icon(Icons.list_alt_rounded, size: 18),
                      label: const Text('Chi tiết'),
                    ),
                ],
              ),
            ),
            const SizedBox(height: 12),
          ],

          // Camera & Gallery Buttons
          Row(
            children: [
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: _isLoading ? null : () => _pickImage(ImageSource.camera),
                  icon: const Icon(Icons.camera_alt_rounded),
                  label: const Text('Chụp Ảnh Kho'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: primaryColor,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _isLoading ? null : () => _pickImage(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library_rounded),
                  label: const Text('Chọn Từ Thư Viện'),
                  style: OutlinedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  void _showDetailsSheet(BuildContext context) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (ctx) {
        return DraggableScrollableSheet(
          initialChildSize: 0.5,
          minChildSize: 0.3,
          maxChildSize: 0.85,
          expand: false,
          builder: (_, controller) {
            return Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Center(
                    child: Container(
                      width: 40,
                      height: 4,
                      decoration: BoxDecoration(
                        color: Colors.grey.shade400,
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        'Danh Sách ${_recognitions.length} Vật Thể Đã Đếm',
                        style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                      ),
                      IconButton(
                        icon: const Icon(Icons.close_rounded),
                        onPressed: () => Navigator.pop(ctx),
                      ),
                    ],
                  ),
                  const Divider(),
                  Expanded(
                    child: ListView.separated(
                      controller: controller,
                      itemCount: _recognitions.length,
                      separatorBuilder: (_, _) => const Divider(height: 1),
                      itemBuilder: (context, index) {
                        final rec = _recognitions[index];
                        return ListTile(
                          leading: CircleAvatar(
                            backgroundColor: Colors.blue.shade100,
                            child: Text(
                              '#${index + 1}',
                              style: TextStyle(
                                color: Colors.blue.shade900,
                                fontWeight: FontWeight.bold,
                                fontSize: 13,
                              ),
                            ),
                          ),
                          title: Text(
                            rec.label,
                            style: const TextStyle(fontWeight: FontWeight.w600),
                          ),
                          subtitle: Text(
                            'Vị trí: [L:${(rec.location.left * 100).toStringAsFixed(0)}%, T:${(rec.location.top * 100).toStringAsFixed(0)}%]',
                            style: TextStyle(fontSize: 11, color: Colors.grey.shade600),
                          ),
                          trailing: Container(
                            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                            decoration: BoxDecoration(
                              color: Colors.green.shade50,
                              borderRadius: BorderRadius.circular(10),
                              border: Border.all(color: Colors.green.shade300),
                            ),
                            child: Text(
                              '${(rec.score * 100).toStringAsFixed(1)}%',
                              style: TextStyle(
                                fontWeight: FontWeight.bold,
                                color: Colors.green.shade900,
                                fontSize: 12,
                              ),
                            ),
                          ),
                        );
                      },
                    ),
                  ),
                ],
              ),
            );
          },
        );
      },
    );
  }

  void _showHistorySheet(BuildContext context) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (ctx) {
        return DraggableScrollableSheet(
          initialChildSize: 0.6,
          minChildSize: 0.3,
          maxChildSize: 0.9,
          expand: false,
          builder: (_, controller) {
            return Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Center(
                    child: Container(
                      width: 40,
                      height: 4,
                      decoration: BoxDecoration(
                        color: Colors.grey.shade400,
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text(
                        'Lịch Sử Đếm Kho Hàng',
                        style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                      IconButton(
                        icon: const Icon(Icons.close_rounded),
                        onPressed: () => Navigator.pop(ctx),
                      ),
                    ],
                  ),
                  const Divider(),
                  if (_scanHistory.isEmpty)
                    const Expanded(
                      child: Center(
                        child: Text(
                          'Chưa có dữ liệu kiểm đếm nào trong phiên làm việc.',
                          style: TextStyle(color: Colors.grey),
                        ),
                      ),
                    )
                  else
                    Expanded(
                      child: ListView.separated(
                        controller: controller,
                        itemCount: _scanHistory.length,
                        separatorBuilder: (_, _) => const SizedBox(height: 12),
                        itemBuilder: (context, index) {
                          final item = _scanHistory[index];
                          return Card(
                            elevation: 0,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                              side: BorderSide(color: Colors.grey.shade300),
                            ),
                            child: Padding(
                              padding: const EdgeInsets.all(10.0),
                              child: Row(
                                children: [
                                  ClipRRect(
                                    borderRadius: BorderRadius.circular(8),
                                    child: Image.memory(
                                      item.imageBytes,
                                      width: 60,
                                      height: 60,
                                      fit: BoxFit.cover,
                                    ),
                                  ),
                                  const SizedBox(width: 12),
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          '${item.timestamp.hour.toString().padLeft(2, '0')}:${item.timestamp.minute.toString().padLeft(2, '0')} - ${item.timestamp.day}/${item.timestamp.month}/${item.timestamp.year}',
                                          style: TextStyle(
                                            fontSize: 12,
                                            color: Colors.grey.shade600,
                                          ),
                                        ),
                                        const SizedBox(height: 2),
                                        Text(
                                          'Chế độ: ${item.filterType}',
                                          style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500),
                                        ),
                                      ],
                                    ),
                                  ),
                                  Container(
                                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                                    decoration: BoxDecoration(
                                      color: Colors.blue.shade50,
                                      borderRadius: BorderRadius.circular(8),
                                    ),
                                    child: Column(
                                      children: [
                                        Text(
                                          '${item.count}',
                                          style: TextStyle(
                                            fontSize: 18,
                                            fontWeight: FontWeight.bold,
                                            color: Colors.blue.shade900,
                                          ),
                                        ),
                                        Text(
                                          'chai/lọ',
                                          style: TextStyle(
                                            fontSize: 10,
                                            color: Colors.blue.shade900,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          );
                        },
                      ),
                    ),
                ],
              ),
            );
          },
        );
      },
    );
  }
}
