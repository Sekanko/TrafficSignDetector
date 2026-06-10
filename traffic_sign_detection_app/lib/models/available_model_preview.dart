class AvailableModelPreview {
  const AvailableModelPreview({
    required this.name,
    required this.version,
    required this.path,
    required this.isDownloaded,
    required this.isSelected,
  });

  final String name;
  final String version;
  final String path;
  final bool isDownloaded;
  final bool isSelected;

  AvailableModelPreview copyWith({
    String? name,
    String? version,
    String? path,
    bool? isDownloaded,
    bool? isSelected,
  }) {
    return AvailableModelPreview(
      name: name ?? this.name,
      version: version ?? this.version,
      path: path ?? this.path,
      isDownloaded: isDownloaded ?? this.isDownloaded,
      isSelected: isSelected ?? this.isSelected,
    );
  }
}
