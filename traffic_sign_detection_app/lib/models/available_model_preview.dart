class AvailableModelPreview {
  const AvailableModelPreview({
    required this.name,
    required this.version,
    required this.path,
    required this.isDownloaded,
    required this.isSelected,
    this.downloadUrl,
    this.modelType = 'single_tflite',
    this.modelFiles = const {},
    this.labels = const [],
    this.inputSize = 320,
    this.confidenceThreshold = 0.45,
    this.isAvailableInApi = true,
    this.hasUpdate = false,
  });

  final String name;
  final String version;
  final String path;
  final bool isDownloaded;
  final bool isSelected;
  final String? downloadUrl;
  final String modelType;
  final Map<String, AvailableModelFile> modelFiles;
  final List<String> labels;
  final int inputSize;
  final double confidenceThreshold;
  final bool isAvailableInApi;
  final bool hasUpdate;

  bool get isBundledAsset => path.startsWith('assets/');

  bool get isPipeline => modelType == 'detector_classifier_pipeline';

  bool get canDownloadFromApi {
    if ((isDownloaded && !hasUpdate) || !isAvailableInApi) {
      return false;
    }

    if (modelFiles.isNotEmpty) {
      return modelFiles.values.every((file) => file.downloadUrl.isNotEmpty);
    }

    return downloadUrl != null && downloadUrl!.isNotEmpty;
  }

  AvailableModelPreview copyWith({
    String? name,
    String? version,
    String? path,
    bool? isDownloaded,
    bool? isSelected,
    String? downloadUrl,
    String? modelType,
    Map<String, AvailableModelFile>? modelFiles,
    List<String>? labels,
    int? inputSize,
    double? confidenceThreshold,
    bool? isAvailableInApi,
    bool? hasUpdate,
  }) {
    return AvailableModelPreview(
      name: name ?? this.name,
      version: version ?? this.version,
      path: path ?? this.path,
      isDownloaded: isDownloaded ?? this.isDownloaded,
      isSelected: isSelected ?? this.isSelected,
      downloadUrl: downloadUrl ?? this.downloadUrl,
      modelType: modelType ?? this.modelType,
      modelFiles: modelFiles ?? this.modelFiles,
      labels: labels ?? this.labels,
      inputSize: inputSize ?? this.inputSize,
      confidenceThreshold: confidenceThreshold ?? this.confidenceThreshold,
      isAvailableInApi: isAvailableInApi ?? this.isAvailableInApi,
      hasUpdate: hasUpdate ?? this.hasUpdate,
    );
  }

  factory AvailableModelPreview.fromJson(Map<String, dynamic> json) {
    final downloadUrl = json['downloadUrl'] ?? json['download_url'];
    final isDownloaded = json['isDownloaded'] ?? json['is_downloaded'];
    final isSelected = json['isSelected'] ?? json['is_selected'];
    final inputSize = json['inputSize'] ?? json['input_size'];
    final confidenceThreshold =
        json['confidenceThreshold'] ?? json['confidence_threshold'];
    final isAvailableInApi =
        json['isAvailableInApi'] ?? json['is_available_in_api'];
    final hasUpdate = json['hasUpdate'] ?? json['has_update'];
    final modelType = json['modelType'] ?? json['model_type'];
    final modelFiles = json['modelFiles'] ?? json['model_files'];

    return AvailableModelPreview(
      name: json['name'] as String? ?? 'Traffic sign model',
      version: json['version'] as String? ?? 'v1.0.0',
      path: json['path'] as String? ?? '',
      downloadUrl: downloadUrl as String?,
      modelType: modelType as String? ?? 'single_tflite',
      modelFiles: _parseModelFiles(modelFiles),
      isDownloaded: isDownloaded as bool? ?? false,
      isSelected: isSelected as bool? ?? false,
      labels: (json['labels'] as List<dynamic>? ?? const [])
          .map((label) => label.toString())
          .toList(),
      inputSize: (inputSize as num?)?.toInt() ?? 320,
      confidenceThreshold: (confidenceThreshold as num?)?.toDouble() ?? 0.45,
      isAvailableInApi: isAvailableInApi as bool? ?? true,
      hasUpdate: hasUpdate as bool? ?? false,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'name': name,
      'version': version,
      'path': path,
      'downloadUrl': downloadUrl,
      'modelType': modelType,
      'modelFiles': modelFiles.map(
        (role, file) => MapEntry(role, file.toJson()),
      ),
      'isDownloaded': isDownloaded,
      'isSelected': isSelected,
      'labels': labels,
      'inputSize': inputSize,
      'confidenceThreshold': confidenceThreshold,
      'isAvailableInApi': isAvailableInApi,
      'hasUpdate': hasUpdate,
    };
  }

  static Map<String, AvailableModelFile> _parseModelFiles(Object? rawFiles) {
    if (rawFiles is! Map<String, dynamic>) {
      return const {};
    }

    return rawFiles.map(
      (role, rawFile) => MapEntry(
        role,
        AvailableModelFile.fromJson(rawFile as Map<String, dynamic>? ?? {}),
      ),
    );
  }
}

class AvailableModelFile {
  const AvailableModelFile({
    required this.path,
    required this.downloadUrl,
    this.inputSize,
    this.outputFormat,
  });

  final String path;
  final String downloadUrl;
  final int? inputSize;
  final String? outputFormat;

  AvailableModelFile copyWith({
    String? path,
    String? downloadUrl,
    int? inputSize,
    String? outputFormat,
  }) {
    return AvailableModelFile(
      path: path ?? this.path,
      downloadUrl: downloadUrl ?? this.downloadUrl,
      inputSize: inputSize ?? this.inputSize,
      outputFormat: outputFormat ?? this.outputFormat,
    );
  }

  factory AvailableModelFile.fromJson(Map<String, dynamic> json) {
    final downloadUrl = json['downloadUrl'] ?? json['download_url'];
    final inputSize = json['inputSize'] ?? json['input_size'];
    final outputFormat = json['outputFormat'] ?? json['output_format'];

    return AvailableModelFile(
      path: json['path'] as String? ?? '',
      downloadUrl: downloadUrl as String? ?? '',
      inputSize: (inputSize as num?)?.toInt(),
      outputFormat: outputFormat as String?,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'path': path,
      'downloadUrl': downloadUrl,
      'inputSize': inputSize,
      'outputFormat': outputFormat,
    };
  }
}
