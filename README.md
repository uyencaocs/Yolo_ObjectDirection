# 📦 YOLO Kho (YOLO Inventory Counter)

![Flutter](https://img.shields.io/badge/Flutter-%2302569B.svg?style=for-the-badge&logo=Flutter&logoColor=white)
![Dart](https://img.shields.io/badge/dart-%230077B5.svg?style=for-the-badge&logo=dart&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Android](https://img.shields.io/badge/Android-3DDC84?style=for-the-badge&logo=android&logoColor=white)
![iOS](https://img.shields.io/badge/iOS-000000?style=for-the-badge&logo=ios&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)

**YOLO Kho** là ứng dụng di động và đa nền tảng được phát triển trên nền tảng **Flutter**, tích hợp mô hình trí tuệ nhân tạo **YOLOv8 (TensorFlow Lite)** cho phép tự động phát hiện, khoanh vùng và kiểm đếm số lượng hàng hóa (chai, lọ, ly, cốc, bình, thùng...) trong kho bãi một cách nhanh chóng, chính xác và trực quan.

---

## 📑 Mục Lục
1. [Giới Thiệu & Tính Năng Nổi Bật](#-giới-thiệu--tính-năng-nổi-bật)
2. [Cấu Trúc Dự Án](#-cấu-trúc-dự-án)
3. [Hướng Dẫn Cài Đặt](#-hướng-dẫn-cài-đặt)
4. [Hướng Dẫn Chạy Ứng Dụng](#-hướng-dẫn-chạy-ứng-dụng)
5. [Hướng Phát Triển (Roadmap)](#-hướng-phát-triển-roadmap)
6. [Đóng Góp & Giấy Phép](#-đóng-góp--giấy-phép)

---

## 🌟 Giới Thiệu & Tính Năng Nổi Bật

### 🎯 Mục tiêu dự án
Trong công tác quản lý kho hàng, việc kiểm đếm thủ công các mặt hàng có số lượng lớn như chai lọ, đồ uống, bao bì... thường tốn nhiều thời gian và dễ xảy ra sai sót. **YOLO Kho** giải quyết bài toán này bằng công nghệ **Computer Vision On-Device** (suy luận trực tiếp trên thiết bị), giúp nhân viên kho chụp ảnh và có ngay kết quả số lượng hàng hóa chỉ trong vài giây.

### ✨ Các tính năng chính
- **Chụp ảnh & Chọn từ thư viện**: Hỗ trợ chụp trực tiếp từ Camera hoặc tải hình ảnh có sẵn từ thiết bị.
- **Suy luận trực tiếp trên thiết bị (On-Device AI Inference)**: Sử dụng mô hình `yolov8n.tflite` qua `tflite_flutter`, hoạt động hoàn toàn **Offline**, không cần kết nối internet hay gửi ảnh lên server ngoại vi, đảm bảo tính bảo mật và tốc độ cao.
- **Tiền xử lý ảnh chuẩn Letterbox**: Giữ nguyên tỉ lệ khung hình (Aspect Ratio) của ảnh chụp, tránh hiện tượng méo ảnh khi đưa vào đầu vào 640x640 của mô hình.
- **Thuật toán NMS (Non-Maximum Suppression)**: Loại bỏ các khung trùng lặp, đảm bảo mỗi vật thể chỉ được đếm duy nhất 1 lần.
- **Đa dạng chế độ lọc kiểm kê**:
  - 🍾 **Chỉ Chai / Lọ**: Tối ưu hóa kiểm đếm các loại chai thủy tinh, chai nhựa, lon nước ngọt (Class ID: 39).
  - 🥛 **Chai, Ly, Cốc & Lọ**: Mở rộng đếm các vật chứa đồ uống như ly rượu, cốc, tách, bát, lọ hoa (Class ID: 39, 40, 41, 45, 75).
  - 📦 **Tất cả vật thể**: Hỗ trợ nhận diện toàn bộ 80 lớp vật thể theo bộ dữ liệu COCO đã được việt hóa 100%.
- **Giao diện hiển thị trực quan (Material Design 3)**:
  - Đánh số thứ tự badge trực tiếp lên từng vật thể được phát hiện.
  - Bounding box màu sắc nét kèm nhãn Tiếng Việt và tỉ lệ phần trăm độ tin cậy (Confidence score).
  - Thẻ thống kê tổng số lượng vật thể đếm được.
- **Lịch sử kiểm kê (Scan History)**: Lưu trữ các lượt quét gần nhất kèm hình ảnh xem trước, số lượng và mốc thời gian để tiện đối soát.

---

## 📁 Cấu Trúc Dự Án

```text
yolo_kho/
├── assets/
│   ├── labels.txt           # Danh sách 80 nhãn vật thể (COCO dataset)
│   └── yolov8n.tflite       # Mô hình YOLOv8 Nano định dạng TensorFlow Lite (Float32)
├── lib/
│   ├── main.dart            # Giao diện chính (UI), quản lý trạng thái, chọn ảnh & lịch sử
│   ├── yolo_detector.dart   # Logic nạp mô hình TFLite, tiền xử lý Letterbox, NMS & suy luận
│   └── box_painter.dart     # CustomPainter vẽ bounding box, badge số thứ tự & nhãn tiếng Việt
├── android/                 # Cấu hình dự án cho nền tảng Android
├── ios/                     # Cấu hình dự án cho nền tảng iOS
├── windows/                 # Cấu hình dự án cho nền tảng Windows Desktop
├── pubspec.yaml             # Khai báo phụ thuộc (dependencies) & tài nguyên assets
└── README.md                # Tài liệu hướng dẫn sử dụng & phát triển
```

---

## 🛠️ Hướng Dẫn Cài Đặt

### 1. Yêu cầu hệ thống (Prerequisites)
Trước khi bắt đầu, hãy đảm bảo máy tính của bạn đã cài đặt:
- **Flutter SDK**: Phiên bản `>= 3.13.3` (Khuyến nghị Flutter `3.19.x` trở lên).
- **Dart SDK**: Phiên bản `>= 3.1.0`.
- **Môi trường phát triển**:
  - Visual Studio Code (với Flutter & Dart extensions) hoặc Android Studio.
  - **Dành cho Android**: Android Studio, Android SDK (API level 21 / Android 5.0 trở lên).
  - **Dành cho iOS**: macOS, Xcode 14+ và CocoaPods.
  - **Dành cho Windows**: Visual Studio 2022 với C++ Build Tools.

### 2. Các bước cài đặt chi tiết

**Bước 1: Tải mã nguồn về máy**
```bash
git clone https://github.com/uyencaocs/Yolo_ObjectDirection.git
cd yolo_kho
```

**Bước 2: Cài đặt các gói phụ thuộc (Dependencies)**
Chạy lệnh sau trong terminal để tải các thư viện cần thiết:
```bash
flutter pub get
```

**Bước 3: Kiểm tra tệp tài nguyên (Assets)**
Đảm bảo thư mục `assets/` đã có đủ 2 tệp:
- `assets/yolov8n.tflite`
- `assets/labels.txt`

Nếu chưa có hoặc muốn cập nhật mô hình mới, bạn có thể xuất mô hình YOLOv8 sang TFLite bằng lệnh Ultralytics Python:
```bash
yolo export model=yolov8n.pt format=tflite imgsz=640
```

**Bước 4: Kiểm tra môi trường Flutter**
```bash
flutter doctor
```
Hãy đảm bảo các dấu tích xanh biểu thị môi trường đã sẵn sàng.

---

## 🚀 Hướng Dẫn Chạy Ứng Dụng

### 1. Khởi chạy ở chế độ Phát triển (Development / Debug)

**Bước 1: Kết nối thiết bị**
- Kết nối điện thoại thật (bật USB Debugging trên Android hoặc Developer Mode trên iOS) **HOẶC** bật thiết bị giả lập (Android Emulator / iOS Simulator).
- Kiểm tra danh sách thiết bị nhận diện:
```bash
flutter devices
```

**Bước 2: Chạy ứng dụng**
- Chạy trên thiết bị Android:
  ```bash
  flutter run -d android
  ```
- Chạy trên thiết bị iOS:
  ```bash
  flutter run -d ios
  ```
- Chạy trên máy tính Windows:
  ```bash
  flutter run -d windows
  ```

---

### 2. Quy trình sử dụng ứng dụng để kiểm kê kho

1. **Khởi động ứng dụng**: Khi mở app `YOLO Kho`, mô hình AI `yolov8n.tflite` sẽ tự động khởi tạo và tải vào bộ nhớ trong vài giây.
2. **Chọn chế độ lọc**:
   - Chọn tab/nút lọc tương ứng: **Chỉ Chai / Lọ**, **Chai, Ly, Cốc & Lọ**, hoặc **Tất cả vật thể**.
3. **Tải ảnh kiểm kê**:
   - Nhấn **Chụp Ảnh** để chụp lại khu vực kệ hàng/lô hàng cần đếm.
   - Hoặc nhấn **Chọn Từ Thư Viện** để tải ảnh chụp kho có sẵn.
4. **Xem kết quả kiểm đếm**:
   - Màn hình sẽ hiển thị bức ảnh kèm các **khung vuông (Bounding Box)** màu sắc nổi bật xung quanh từng sản phẩm.
   - Mỗi sản phẩm được gắn **Badge số thứ tự (1, 2, 3...)** ở góc trên bên trái khung.
   - Thẻ kết quả phía dưới hiển thị **Tổng số lượng phát hiện** (Ví dụ: `Đã đếm: 12 Chai/Lọ`).
5. **Xem lịch sử quét**:
   - Vuốt xuống phần **Lịch sử kiểm kê** để xem lại danh sách các lần đếm trước đó kèm ảnh thu nhỏ, thời gian và số lượng chi tiết.

---

### 3. Biên dịch bản Thương mại (Production Release Build)

Khi muốn đóng gói ứng dụng để cài đặt chính thức:

- **Xuất file APK Android**:
  ```bash
  flutter build apk --release
  ```
  *File APK đầu ra nằm tại:* `build/app/outputs/flutter-apk/app-release.apk`

- **Xuất file Android App Bundle (AAB) để đưa lên Google Play**:
  ```bash
  flutter build appbundle --release
  ```

- **Xuất gói iOS (IPA)**:
  ```bash
  flutter build ipa --release
  ```

- **Xuất ứng dụng Windows Desktop**:
  ```bash
  flutter build windows --release
  ```
  *Thư mục ứng dụng nằm tại:* `build/windows/runner/Release/`

---

## 🔮 Hướng Phát Triển (Roadmap)

Dự án **YOLO Kho** hướng tới việc trở thành một giải pháp kiểm kê kho hàng thông minh, toàn diện cho các doanh nghiệp vừa và nhỏ (SME), chuỗi bán lẻ và nhà kho logistics. Dưới đây là các định hướng nâng cấp trong tương lai:

### 1. Fine-Tune & Đào Tạo Mô Hình Chuyên Biệt
- [ ] **Huấn luyện mô hình YOLOv8 Custom**: Train trên tập dữ liệu kho thực tế tại Việt Nam với các sản phẩm đặc thù (chai bia, lon nước ngọt, chai dầu ăn, chai nước mắm, thùng carton, v.v.) nhằm đạt độ chính xác **> 98%**.
- [ ] **Hỗ trợ đọc mã vạch (Barcode) / QR Code**: Kết hợp nhận diện vị trí sản phẩm và tự động quét mã vạch sản phẩm song song.

### 2. Đếm Thời Gian Thực Qua Video Stream (Real-Time Live Camera Feed)
- [ ] **Tích hợp Camera Stream**: Cho phép người dùng quét camera liên tục qua kệ hàng thay vì phải chụp từng bức ảnh tĩnh.
- [ ] **Theo dõi vật thể (Object Tracking)**: Tích hợp các thuật toán như **BYTETracker** hoặc **DeepSORT** giúp theo dõi ID vật thể khi camera di chuyển, tránh đếm lặp lại các sản phẩm đã đi qua khung hình.

### 3. Quản Lý Dữ Liệu & Đồng Bộ Cloud / ERP
- [ ] **Xuất báo cáo đa định dạng**: Cho phép xuất danh sách kiểm kê ra các tệp **Excel (.xlsx)**, **PDF** hoặc **CSV** có chèn hình ảnh minh họa.
- [ ] **Kết nối API Backend & WMS/ERP**: Đồng bộ dữ liệu kiểm kê tức thì về hệ thống quản lý kho trung tâm (như Odoo, SAP, ERP Custom).

### 4. Tối Ưu Hóa Hiệu Năng & AI Edge Processing
- [ ] **Tối ưu hóa mô hình TFLite (INT8 Quantization)**: Nén kích thước mô hình từ ~12MB xuống còn ~3MB, giúp giảm 70% dung lượng bộ nhớ RAM tiêu thụ và tăng tốc độ suy luận gấp 3 lần trên các máy cấu hình thấp.
- [ ] **Tăng tốc phần cứng (Hardware Acceleration)**: Kích hoạt GPU Delegate / NNAPI (Android) và Metal Delegate (iOS) để suy luận thời gian thực mượt mà ở tốc độ 30+ FPS.

---

## 📝 Đóng Góp & Giấy Phép

### Đóng góp (Contributing)
Mọi đóng góp từ cộng đồng đều được hoan nghênh! Nếu bạn muốn cải tiến dự án:
1. Fork repository này.
2. Tạo branch tính năng mới (`git checkout -b feature/AmazingFeature`).
3. Commit các thay đổi (`git commit -m 'Add some AmazingFeature'`).
4. Push lên branch (`git push origin feature/AmazingFeature`).
5. Mở một **Pull Request**.

### Giấy phép (License)
Dự án được phát hành dưới giấy phép **MIT License**. Bạn có thể tự do sử dụng, chỉnh sửa và phân phối cho mục đích cá nhân hoặc thương mại.

---

<p center align="center">
  <i>Được phát triển với ❤️ sử dụng Flutter & YOLOv8</i>
</p>
