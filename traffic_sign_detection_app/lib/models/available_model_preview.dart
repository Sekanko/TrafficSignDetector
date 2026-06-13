class AvailableModelPreview {
  const AvailableModelPreview({
    required this.name,
    required this.version,
    required this.path,
    required this.isDownloaded,
    required this.isSelected,
    this.downloadUrl,
    this.labels = const [],
    this.inputSize = 320,
    this.confidenceThreshold = 0.45,
    this.isAvailableInApi = true,
  });

  final String name;
  final String version;
  final String path;
  final bool isDownloaded;
  final bool isSelected;
  final String? downloadUrl;
  final List<String> labels;
  final int inputSize;
  final double confidenceThreshold;
  final bool isAvailableInApi;

  bool get isBundledAsset => path.startsWith('assets/');

  AvailableModelPreview copyWith({
    String? name,
    String? version,
    String? path,
    bool? isDownloaded,
    bool? isSelected,
    String? downloadUrl,
    List<String>? labels,
    int? inputSize,
    double? confidenceThreshold,
    bool? isAvailableInApi,
  }) {
    return AvailableModelPreview(
      name: name ?? this.name,
      version: version ?? this.version,
      path: path ?? this.path,
      isDownloaded: isDownloaded ?? this.isDownloaded,
      isSelected: isSelected ?? this.isSelected,
      downloadUrl: downloadUrl ?? this.downloadUrl,
      labels: labels ?? this.labels,
      inputSize: inputSize ?? this.inputSize,
      confidenceThreshold: confidenceThreshold ?? this.confidenceThreshold,
      isAvailableInApi: isAvailableInApi ?? this.isAvailableInApi,
    );
  }

  factory AvailableModelPreview.fromJson(Map<String, dynamic> json) {
    return AvailableModelPreview(
      name: json['name'] as String? ?? 'Traffic sign model',
      version: json['version'] as String? ?? 'v1.0.0',
      path: json['path'] as String? ?? '',
      downloadUrl: json['downloadUrl'] as String?,
      isDownloaded: json['isDownloaded'] as bool? ?? false,
      isSelected: json['isSelected'] as bool? ?? false,
      labels: (json['labels'] as List<dynamic>? ?? const [])
          .map((label) => label.toString())
          .toList(),
      inputSize: json['inputSize'] as int? ?? 320,
      confidenceThreshold:
          (json['confidenceThreshold'] as num?)?.toDouble() ?? 0.45,
      isAvailableInApi: json['isAvailableInApi'] as bool? ?? true,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'name': name,
      'version': version,
      'path': path,
      'downloadUrl': downloadUrl,
      'isDownloaded': isDownloaded,
      'isSelected': isSelected,
      'labels': labels,
      'inputSize': inputSize,
      'confidenceThreshold': confidenceThreshold,
      'isAvailableInApi': isAvailableInApi,
    };
  }
}
