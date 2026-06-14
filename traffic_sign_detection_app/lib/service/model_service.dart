import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:ui';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/available_model_preview.dart';
import '../models/detected_sign_preview.dart';
import 'detection_output_parser.dart';
import 'model_input_converter.dart';

class ModelService extends ChangeNotifier {
  ModelService._();

  static final ModelService instance = ModelService._();
  static const _configuredModelsApiUrl = String.fromEnvironment(
    'MODELS_API_URL',
  );

  static Uri get defaultModelsApiUri {
    if (_configuredModelsApiUrl.isNotEmpty) {
      return Uri.parse(_configuredModelsApiUrl);
    }

    // Physical Android + `adb reverse tcp:8000 tcp:8000` reaches the host via
    // 127.0.0.1. Android emulator needs:
    // --dart-define=MODELS_API_URL=http://10.0.2.2:8000/models
    final host = Platform.isAndroid ? '127.0.0.1' : 'localhost';
    return Uri.parse('http://$host:8000/models');
  }

  static const _apiTimeout = Duration(seconds: 15);
  static const _removedModelNames = {'Traffic Sign YOLO All'};

  final HttpClient _httpClient = HttpClient()..connectionTimeout = _apiTimeout;

  Interpreter? _detectorInterpreter;
  Interpreter? _classifierInterpreter;
  AvailableModelPreview? _currentModel;
  List<AvailableModelPreview> _availableModels = const [];
  bool _isInitialized = false;
  bool _isBusy = false;
  String? _lastError;

  List<AvailableModelPreview> get availableModels =>
      List.unmodifiable(_availableModels);

  AvailableModelPreview? get currentModel => _currentModel;

  bool get hasLoadedModel =>
      _detectorInterpreter != null && _classifierInterpreter != null;

  bool get isBusy => _isBusy;

  String? get lastError => _lastError;

  Future<void> initialize({Uri? modelsApiUri}) async {
    if (_isInitialized) {
      return;
    }

    _setBusy(true);
    try {
      await fetchAvailableModels(modelsApiUri ?? defaultModelsApiUri);

      final selectedModel = _availableModels
          .cast<AvailableModelPreview?>()
          .firstWhere(
            (model) => model?.isSelected == true && model?.isDownloaded == true,
            orElse: () =>
                _availableModels.cast<AvailableModelPreview?>().firstWhere(
                  (model) => model?.isDownloaded == true,
                  orElse: () => null,
                ),
          );

      if (selectedModel != null) {
        await loadModel(selectedModel);
      }

      _isInitialized = true;
    } catch (error) {
      _lastError = 'Nie udało się zainicjalizować modeli: $error';
    } finally {
      _setBusy(false);
    }
  }

  Future<void> fetchAvailableModels(Uri apiUri) async {
    var apiModels = <AvailableModelPreview>[];

    try {
      apiModels = await _fetchModelsFromApi(apiUri);
      _lastError = null;
    } catch (error) {
      _lastError = 'Nie udało się pobrać listy modeli z API ($apiUri): $error';
    }

    final localModels = await _loadDownloadedModelsRegistry();
    _availableModels = await _markDownloadedModels(
      _mergeApiAndLocalModels(
        _supportedModels(apiModels),
        _supportedModels(localModels),
      ),
    );
    notifyListeners();
  }

  Future<List<AvailableModelPreview>> _fetchModelsFromApi(Uri apiUri) async {
    final request = await _httpClient.getUrl(apiUri).timeout(_apiTimeout);
    final response = await request.close().timeout(_apiTimeout);

    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw HttpException(
        'API modeli zwróciło status ${response.statusCode}',
        uri: apiUri,
      );
    }

    final body = await response
        .transform(utf8.decoder)
        .join()
        .timeout(_apiTimeout);
    final decoded = jsonDecode(body);
    final rawModels = decoded is List<dynamic>
        ? decoded
        : (decoded as Map<String, dynamic>)['models'] as List<dynamic>? ??
              const [];

    return rawModels
        .whereType<Map<String, dynamic>>()
        .map(AvailableModelPreview.fromJson)
        .map((model) => model.copyWith(isAvailableInApi: true))
        .toList();
  }

  Future<void> downloadModel(AvailableModelPreview model) async {
    if (!model.isPipeline || _isRemovedModel(model)) {
      _lastError =
          'Aplikacja obsługuje tylko pipeline detektor + klasyfikator.';
      notifyListeners();
      return;
    }

    if (model.modelFiles.isNotEmpty) {
      await _downloadModelBundle(model);
      return;
    }

    if (model.downloadUrl == null || model.downloadUrl!.isEmpty) {
      _lastError =
          'Model ${model.name} nie ma skonfigurowanego adresu pobrania.';
      notifyListeners();
      return;
    }

    _setBusy(true);
    try {
      final uri = Uri.parse(model.downloadUrl!);
      final request = await _httpClient.getUrl(uri);
      final response = await request.close();

      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw HttpException(
          'Pobieranie modelu zwróciło status ${response.statusCode}',
          uri: uri,
        );
      }

      final bytes = await consolidateHttpClientResponseBytes(response);
      final modelsDirectory = await _modelsDirectory();
      final file = File('${modelsDirectory.path}/${_modelFileName(model)}');
      await file.writeAsBytes(bytes, flush: true);

      final downloadedModel = model.copyWith(
        path: file.path,
        isDownloaded: true,
        isAvailableInApi: true,
        hasUpdate: false,
      );
      _replaceModel(model, downloadedModel);
      await _saveDownloadedModelToRegistry(downloadedModel);
      _lastError = null;
    } catch (error) {
      _lastError = 'Nie udało się pobrać modelu ${model.name}: $error';
    } finally {
      _setBusy(false);
    }
  }

  Future<void> _downloadModelBundle(AvailableModelPreview model) async {
    final missingDownload = model.modelFiles.entries
        .where((entry) => entry.value.downloadUrl.isEmpty)
        .map((entry) => entry.key)
        .toList();
    if (missingDownload.isNotEmpty) {
      _lastError =
          'Model ${model.name} nie ma adresów pobrania dla: ${missingDownload.join(', ')}.';
      notifyListeners();
      return;
    }

    _setBusy(true);
    try {
      final bundleDirectory = await _modelBundleDirectory(model);
      final downloadedFiles = <String, AvailableModelFile>{};

      for (final entry in model.modelFiles.entries) {
        final fileMetadata = entry.value;
        final localFile = File(
          '${bundleDirectory.path}/${_safeFileName(fileMetadata.path)}',
        );
        await _downloadToFile(Uri.parse(fileMetadata.downloadUrl), localFile);
        downloadedFiles[entry.key] = fileMetadata.copyWith(
          path: localFile.path,
        );
      }

      final downloadedModel = model.copyWith(
        path: bundleDirectory.path,
        modelFiles: downloadedFiles,
        isDownloaded: true,
        isAvailableInApi: true,
        hasUpdate: false,
      );
      _replaceModel(model, downloadedModel);
      await _saveDownloadedModelToRegistry(downloadedModel);
      await loadModel(downloadedModel);
      _lastError = null;
    } catch (error) {
      _lastError = 'Nie udało się pobrać modelu ${model.name}: $error';
    } finally {
      _setBusy(false);
    }
  }

  Future<void> _downloadToFile(Uri uri, File file) async {
    final request = await _httpClient.getUrl(uri);
    final response = await request.close();

    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw HttpException(
        'Pobieranie modelu zwróciło status ${response.statusCode}',
        uri: uri,
      );
    }

    final bytes = await consolidateHttpClientResponseBytes(response);
    if (!await file.parent.exists()) {
      await file.parent.create(recursive: true);
    }
    await file.writeAsBytes(bytes, flush: true);
  }

  Future<void> deleteModel(AvailableModelPreview model) async {
    if (model.isBundledAsset) {
      _lastError = 'Model wbudowany w aplikację nie może zostać usunięty.';
      notifyListeners();
      return;
    }

    _setBusy(true);
    try {
      if (model.modelFiles.isNotEmpty) {
        final directory = Directory(model.path);
        if (await directory.exists()) {
          await directory.delete(recursive: true);
        }
      } else {
        final file = File(model.path);
        if (await file.exists()) {
          await file.delete();
        }
      }

      if (_currentModel != null && _isSameModel(_currentModel!, model)) {
        _closeInterpreters();
        _currentModel = null;
      }

      await _removeDownloadedModelFromRegistry(model);

      if (model.isAvailableInApi) {
        _replaceModel(
          model,
          model.copyWith(isDownloaded: false, isSelected: false),
        );
      } else {
        _availableModels = _availableModels
            .where((candidate) => !_isSameModel(candidate, model))
            .map((candidate) => candidate.copyWith(isSelected: false))
            .toList();
        notifyListeners();
      }
      _lastError = null;
    } catch (error) {
      _lastError = 'Nie udało się usunąć modelu ${model.name}: $error';
    } finally {
      _setBusy(false);
    }
  }

  Future<void> loadModel(AvailableModelPreview model) async {
    if (!model.isPipeline || _isRemovedModel(model)) {
      _lastError =
          'Aplikacja obsługuje tylko pipeline detektor + klasyfikator.';
      notifyListeners();
      return;
    }

    if (!model.isDownloaded) {
      await downloadModel(model);
      final downloadedModel = _availableModels.firstWhere(
        (candidate) => _isSameModel(candidate, model),
        orElse: () => model,
      );
      if (!downloadedModel.isDownloaded) {
        return;
      }
      return loadModel(downloadedModel);
    }

    _setBusy(true);
    try {
      await _loadPipelineModel(model);
      _currentModel = model.copyWith(isSelected: true);
      _availableModels = _availableModels
          .map(
            (candidate) =>
                candidate.copyWith(isSelected: _isSameModel(candidate, model)),
          )
          .toList();
      if (model.isDownloaded) {
        await _updateSelectedInRegistry(model);
      }
      _lastError = null;
    } catch (error) {
      _lastError = 'Nie udało się załadować modelu ${model.name}: $error';
    } finally {
      _setBusy(false);
    }
  }

  Future<void> _loadPipelineModel(AvailableModelPreview model) async {
    final detectorFile = model.modelFiles['detector'];
    final classifierFile = model.modelFiles['classifier'];
    if (detectorFile == null || classifierFile == null) {
      throw StateError('Pakiet modelu nie zawiera detektora i klasyfikatora.');
    }

    final nextDetector = Interpreter.fromFile(
      File(detectorFile.path),
      options: _interpreterOptions(),
    );
    final nextClassifier = Interpreter.fromFile(
      File(classifierFile.path),
      options: _interpreterOptions(),
    );

    _closeInterpreters();
    _detectorInterpreter = nextDetector;
    _classifierInterpreter = nextClassifier;
  }

  Future<List<DetectedSignPreview>> predict(
    CameraImage image, {
    int rotationDegrees = 0,
  }) async {
    final model = _currentModel;
    if (model == null) {
      return const [];
    }

    try {
      if (!model.isPipeline) {
        return const [];
      }

      return _predictPipeline(image, model, rotationDegrees: rotationDegrees);
    } catch (error) {
      _lastError = 'Predykcja nie powiodła się: $error';
      notifyListeners();
      return const [];
    }
  }

  Future<List<DetectedSignPreview>> _predictPipeline(
    CameraImage image,
    AvailableModelPreview model, {
    required int rotationDegrees,
  }) async {
    final detector = _detectorInterpreter;
    final classifier = _classifierInterpreter;
    if (detector == null || classifier == null) {
      return const [];
    }

    final detectorInputTensor = detector.getInputTensor(0);
    final detectorInput = ModelInputConverter.cameraImageToInput(
      image: image,
      inputShape: detectorInputTensor.shape,
      fallbackSize: model.modelFiles['detector']?.inputSize ?? model.inputSize,
      type: detectorInputTensor.type,
      rotationDegrees: rotationDegrees,
    );
    if (detectorInput == null) {
      return const [];
    }

    final detectorOutputs = _emptyOutputs(detector);
    detector.runForMultipleInputs([detectorInput], detectorOutputs);

    final candidates = DetectionOutputParser.parseDetectorCandidates(
      detectorOutputs,
      0.20,
    ).take(5).toList();
    if (candidates.isEmpty) {
      return const [];
    }

    final classifierInputTensor = classifier.getInputTensor(0);
    final results = <DetectedSignPreview>[];
    for (final candidate in candidates) {
      final classifierInput = ModelInputConverter.cameraImageCropToInput(
        image: image,
        crop: _expandedRect(candidate.boundingBox, margin: 0.1),
        inputShape: classifierInputTensor.shape,
        fallbackSize: model.modelFiles['classifier']?.inputSize ?? 224,
        type: classifierInputTensor.type,
        rotationDegrees: rotationDegrees,
      );
      if (classifierInput == null) {
        continue;
      }

      final classifierOutputs = _emptyOutputs(classifier);
      classifier.runForMultipleInputs([classifierInput], classifierOutputs);
      final classification = DetectionOutputParser.parseClassification(
        classifierOutputs[0],
      );
      if (classification == null) {
        continue;
      }

      results.add(
        DetectedSignPreview(
          label: _labelForClass(model, classification.$1),
          confidence: classification.$2,
          boundingBox: candidate.boundingBox,
        ),
      );
    }

    return results;
  }

  void disposeService() {
    _closeInterpreters();
    _httpClient.close(force: true);
  }

  Future<List<AvailableModelPreview>> _markDownloadedModels(
    List<AvailableModelPreview> models,
  ) async {
    return Future.wait(
      models.map((model) async {
        if (model.isBundledAsset) {
          return model.copyWith(isDownloaded: true);
        }

        if (model.hasUpdate) {
          return model.copyWith(isDownloaded: true);
        }

        if (model.modelFiles.isNotEmpty) {
          final bundleDirectory = await _modelBundleDirectory(model);
          final localFiles = model.modelFiles.map(
            (role, file) => MapEntry(
              role,
              file.copyWith(
                path: '${bundleDirectory.path}/${_safeFileName(file.path)}',
              ),
            ),
          );
          final isDownloaded = await _allModelFilesExist(localFiles);
          return model.copyWith(
            path: bundleDirectory.path,
            modelFiles: localFiles,
            isDownloaded: isDownloaded,
          );
        }

        final file = File(model.path);
        if (await file.exists()) {
          return model.copyWith(isDownloaded: true);
        }

        if (model.downloadUrl == null) {
          return model.copyWith(isDownloaded: false);
        }

        final modelsDirectory = await _modelsDirectory();
        final localFile = File(
          '${modelsDirectory.path}/${_modelFileName(model)}',
        );
        return model.copyWith(
          path: localFile.path,
          isDownloaded: await localFile.exists(),
        );
      }),
    );
  }

  Future<Directory> _modelsDirectory() async {
    final supportDirectory = await getApplicationSupportDirectory();
    final modelsDirectory = Directory('${supportDirectory.path}/models');
    if (!await modelsDirectory.exists()) {
      await modelsDirectory.create(recursive: true);
    }
    return modelsDirectory;
  }

  Future<Directory> _modelBundleDirectory(AvailableModelPreview model) async {
    final modelsDirectory = await _modelsDirectory();
    final bundleDirectory = Directory(
      '${modelsDirectory.path}/${_safeModelDirectoryName(model)}',
    );
    if (!await bundleDirectory.exists()) {
      await bundleDirectory.create(recursive: true);
    }
    return bundleDirectory;
  }

  Future<File> _registryFile() async {
    final modelsDirectory = await _modelsDirectory();
    return File('${modelsDirectory.path}/downloaded_models.json');
  }

  Future<List<Map<String, dynamic>>> _readRegistryRaw() async {
    final registryFile = await _registryFile();
    if (!await registryFile.exists()) {
      return [];
    }

    try {
      final decoded = jsonDecode(await registryFile.readAsString());
      if (decoded is! List<dynamic>) {
        return [];
      }

      return decoded.whereType<Map<String, dynamic>>().toList();
    } catch (_) {
      return [];
    }
  }

  Future<void> _writeRegistryRaw(List<Map<String, dynamic>> entries) async {
    final registryFile = await _registryFile();
    if (entries.isEmpty) {
      if (await registryFile.exists()) {
        await registryFile.delete();
      }
      return;
    }

    await registryFile.writeAsString(jsonEncode(entries), flush: true);
  }

  Future<List<AvailableModelPreview>> _loadDownloadedModelsRegistry() async {
    final entries = await _readRegistryRaw();
    final models = <AvailableModelPreview>[];

    for (final entry in entries) {
      final model = AvailableModelPreview.fromJson(entry).copyWith(
        isAvailableInApi: false,
        isDownloaded: true,
        downloadUrl: null,
      );
      if (await _modelStorageExists(model)) {
        models.add(model);
      }
    }

    return models;
  }

  Future<void> _saveDownloadedModelToRegistry(
    AvailableModelPreview model,
  ) async {
    final entries = await _readRegistryRaw();
    final updatedEntries = [
      ...entries.where((entry) => entry['name'] != model.name),
      _toRegistryEntry(model.copyWith(isDownloaded: true, isSelected: false)),
    ];
    await _writeRegistryRaw(updatedEntries);
  }

  Future<void> _removeDownloadedModelFromRegistry(
    AvailableModelPreview model,
  ) async {
    final entries = await _readRegistryRaw();
    final updatedEntries = entries
        .where((entry) => entry['name'] != model.name)
        .toList();
    await _writeRegistryRaw(updatedEntries);
  }

  List<AvailableModelPreview> _mergeApiAndLocalModels(
    List<AvailableModelPreview> apiModels,
    List<AvailableModelPreview> localModels,
  ) {
    final merged = <String, AvailableModelPreview>{};

    for (final apiModel in apiModels) {
      merged[_modelKey(apiModel)] = apiModel.copyWith(isAvailableInApi: true);
    }

    for (final localModel in localModels) {
      final key = _modelKey(localModel);
      final existing = merged[key];
      if (existing != null) {
        final mergedFiles = existing.modelFiles.map((role, apiFile) {
          final localFile = localModel.modelFiles[role];
          return MapEntry(
            role,
            apiFile.copyWith(path: localFile?.path ?? apiFile.path),
          );
        });

        if (existing.version != localModel.version) {
          merged[key] = existing.copyWith(
            path: localModel.path,
            modelFiles: mergedFiles.isNotEmpty
                ? mergedFiles
                : localModel.modelFiles,
            isDownloaded: true,
            isSelected: localModel.isSelected,
            hasUpdate: true,
          );
          continue;
        }

        merged[key] = existing.copyWith(
          path: localModel.path,
          isDownloaded: true,
          isSelected: localModel.isSelected || existing.isSelected,
          hasUpdate: false,
          labels: localModel.labels.isNotEmpty
              ? localModel.labels
              : existing.labels,
          modelFiles: mergedFiles.isNotEmpty
              ? mergedFiles
              : localModel.modelFiles,
          inputSize: existing.inputSize,
          confidenceThreshold: existing.confidenceThreshold,
        );
        continue;
      }

      merged[key] = localModel.copyWith(
        isAvailableInApi: false,
        isDownloaded: true,
        downloadUrl: null,
      );
    }

    return merged.values.toList()..sort(_compareModels);
  }

  List<AvailableModelPreview> _supportedModels(
    List<AvailableModelPreview> models,
  ) {
    return models
        .where((model) => model.isPipeline && !_isRemovedModel(model))
        .toList();
  }

  bool _isRemovedModel(AvailableModelPreview model) {
    return _removedModelNames.contains(model.name) ||
        model.path.endsWith('yolo_all.tflite');
  }

  String _modelKey(AvailableModelPreview model) {
    return model.name;
  }

  bool _isSameModel(AvailableModelPreview a, AvailableModelPreview b) {
    return a.name == b.name;
  }

  int _compareModels(AvailableModelPreview a, AvailableModelPreview b) {
    final nameComparison = a.name.compareTo(b.name);
    if (nameComparison != 0) {
      return nameComparison;
    }

    return b.version.compareTo(a.version);
  }

  String _modelFileName(AvailableModelPreview model) {
    final safeName = '${model.name}_${model.version}'
        .toLowerCase()
        .replaceAll(RegExp(r'[^a-z0-9]+'), '_')
        .replaceAll(RegExp(r'_+'), '_')
        .replaceAll(RegExp(r'^_|_$'), '');
    return '$safeName.tflite';
  }

  String _safeModelDirectoryName(AvailableModelPreview model) {
    return '${model.name}_${model.version}'
        .toLowerCase()
        .replaceAll(RegExp(r'[^a-z0-9]+'), '_')
        .replaceAll(RegExp(r'_+'), '_')
        .replaceAll(RegExp(r'^_|_$'), '');
  }

  String _safeFileName(String path) {
    return path.split(Platform.pathSeparator).last.split('/').last;
  }

  Future<bool> _allModelFilesExist(
    Map<String, AvailableModelFile> files,
  ) async {
    if (files.isEmpty) {
      return false;
    }

    for (final file in files.values) {
      if (!await File(file.path).exists()) {
        return false;
      }
    }
    return true;
  }

  Future<bool> _modelStorageExists(AvailableModelPreview model) async {
    if (model.modelFiles.isNotEmpty) {
      return _allModelFilesExist(model.modelFiles);
    }

    return File(model.path).exists();
  }

  Map<int, Object> _emptyOutputs(Interpreter interpreter) {
    final outputTensors = interpreter.getOutputTensors();
    return <int, Object>{
      for (var index = 0; index < outputTensors.length; index++)
        index: ModelInputConverter.emptyTensorData(
          outputTensors[index].shape,
          outputTensors[index].type,
        ),
    };
  }

  Rect _expandedRect(Rect rect, {required double margin}) {
    final dx = rect.width * margin;
    final dy = rect.height * margin;
    return Rect.fromLTRB(
      (rect.left - dx).clamp(0.0, 1.0),
      (rect.top - dy).clamp(0.0, 1.0),
      (rect.right + dx).clamp(0.0, 1.0),
      (rect.bottom + dy).clamp(0.0, 1.0),
    );
  }

  String _labelForClass(AvailableModelPreview model, int classIndex) {
    if (classIndex >= 0 && classIndex < model.labels.length) {
      return model.labels[classIndex];
    }
    return 'Znak ${classIndex + 1}';
  }

  InterpreterOptions _interpreterOptions() {
    return InterpreterOptions()..threads = 2;
  }

  void _closeInterpreters() {
    _detectorInterpreter?.close();
    _classifierInterpreter?.close();
    _detectorInterpreter = null;
    _classifierInterpreter = null;
  }

  Map<String, dynamic> _toRegistryEntry(AvailableModelPreview model) {
    return model
        .copyWith(
          downloadUrl: null,
          isAvailableInApi: false,
          hasUpdate: false,
          modelFiles: model.modelFiles.map(
            (role, file) => MapEntry(role, file.copyWith(downloadUrl: '')),
          ),
        )
        .toJson();
  }

  Future<void> _updateSelectedInRegistry(AvailableModelPreview model) async {
    var entries = await _readRegistryRaw();
    final hasEntry = entries.any((entry) => entry['name'] == model.name);

    if (!hasEntry) {
      entries = [
        ...entries,
        _toRegistryEntry(
          model.copyWith(
            isDownloaded: true,
            isSelected: true,
            downloadUrl: null,
          ),
        ),
      ];
    }

    entries = entries
        .map((entry) => {...entry, 'isSelected': entry['name'] == model.name})
        .toList();
    await _writeRegistryRaw(entries);
  }

  void _replaceModel(
    AvailableModelPreview oldModel,
    AvailableModelPreview newModel,
  ) {
    _availableModels = _availableModels
        .map(
          (candidate) =>
              _isSameModel(candidate, oldModel) ? newModel : candidate,
        )
        .toList();
    notifyListeners();
  }

  void _setBusy(bool value) {
    _isBusy = value;
    notifyListeners();
  }
}
