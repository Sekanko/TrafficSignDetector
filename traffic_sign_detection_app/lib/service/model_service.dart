import 'dart:async';
import 'dart:convert';
import 'dart:io';

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

    final host = Platform.isAndroid ? '10.0.2.2' : 'localhost';
    return Uri.parse('http://$host:8000/models');
  }

  final HttpClient _httpClient = HttpClient()
    ..connectionTimeout = const Duration(seconds: 3);

  Interpreter? _interpreter;
  AvailableModelPreview? _currentModel;
  List<AvailableModelPreview> _availableModels = _fallbackModels;
  bool _isInitialized = false;
  bool _isBusy = false;
  String? _lastError;

  List<AvailableModelPreview> get availableModels =>
      List.unmodifiable(_availableModels);

  AvailableModelPreview? get currentModel => _currentModel;

  bool get hasLoadedModel => _interpreter != null;

  bool get isBusy => _isBusy;

  String? get lastError => _lastError;

  Future<void> initialize({Uri? modelsApiUri}) async {
    if (_isInitialized) {
      return;
    }

    _setBusy(true);
    try {
      if (modelsApiUri == null) {
        final localModels = await _loadDownloadedModelsRegistry();
        _availableModels = await _markDownloadedModels(
          _mergeApiAndLocalModels(_fallbackModels, localModels),
        );
      } else {
        await fetchAvailableModels(modelsApiUri);
      }

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
      _lastError = null;
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
      if (apiModels.isEmpty) {
        apiModels = List.of(_fallbackModels);
      }
      _lastError = null;
    } catch (error) {
      _lastError = 'Nie udało się pobrać listy modeli z API: $error';
    }

    final localModels = await _loadDownloadedModelsRegistry();
    _availableModels = await _markDownloadedModels(
      _mergeApiAndLocalModels(apiModels, localModels),
    );
    notifyListeners();
  }

  Future<List<AvailableModelPreview>> _fetchModelsFromApi(Uri apiUri) async {
    final request = await _httpClient
        .getUrl(apiUri)
        .timeout(const Duration(seconds: 3));
    final response = await request.close().timeout(const Duration(seconds: 3));

    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw HttpException(
        'API modeli zwróciło status ${response.statusCode}',
        uri: apiUri,
      );
    }

    final body = await response
        .transform(utf8.decoder)
        .join()
        .timeout(const Duration(seconds: 3));
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

  Future<void> deleteModel(AvailableModelPreview model) async {
    if (model.isBundledAsset) {
      _lastError = 'Model wbudowany w aplikację nie może zostać usunięty.';
      notifyListeners();
      return;
    }

    _setBusy(true);
    try {
      final file = File(model.path);
      if (await file.exists()) {
        await file.delete();
      }

      if (_currentModel != null && _isSameModel(_currentModel!, model)) {
        _interpreter?.close();
        _interpreter = null;
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
            .map(
              (candidate) => candidate.copyWith(
                isSelected: false,
              ),
            )
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
      final nextInterpreter = model.isBundledAsset
          ? await Interpreter.fromAsset(model.path)
          : Interpreter.fromFile(File(model.path));

      _interpreter?.close();
      _interpreter = nextInterpreter;
      _currentModel = model.copyWith(isSelected: true);
      _availableModels = _availableModels
          .map(
            (candidate) => candidate.copyWith(
              isSelected: _isSameModel(candidate, model),
            ),
          )
          .toList();
      _lastError = null;
    } catch (error) {
      _lastError = 'Nie udało się załadować modelu ${model.name}: $error';
    } finally {
      _setBusy(false);
    }
  }

  Future<List<DetectedSignPreview>> predict(CameraImage image) async {
    final interpreter = _interpreter;
    final model = _currentModel;
    if (interpreter == null || model == null) {
      return const [];
    }

    try {
      final inputTensor = interpreter.getInputTensor(0);
      final inputShape = inputTensor.shape;
      final input = ModelInputConverter.cameraImageToInput(
        image: image,
        inputShape: inputShape,
        fallbackSize: model.inputSize,
        type: inputTensor.type,
      );
      if (input == null) {
        return const [];
      }

      final outputTensors = interpreter.getOutputTensors();
      final outputs = <int, Object>{
        for (var index = 0; index < outputTensors.length; index++)
          index: ModelInputConverter.emptyTensorData(
            outputTensors[index].shape,
            outputTensors[index].type,
          ),
      };

      interpreter.runForMultipleInputs([input], outputs);

      return DetectionOutputParser.parse(outputs, model);
    } catch (error) {
      _lastError = 'Predykcja nie powiodła się: $error';
      notifyListeners();
      return const [];
    }
  }

  void disposeService() {
    _interpreter?.close();
    _interpreter = null;
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

        final file = File(model.path);
        if (await file.exists()) {
          return model.copyWith(isDownloaded: true);
        }

        if (model.downloadUrl == null) {
          return model.copyWith(isDownloaded: false);
        }

        final modelsDirectory = await _modelsDirectory();
        final localFile = File('${modelsDirectory.path}/${_modelFileName(model)}');
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
        isSelected: false,
      );
      if (await File(model.path).exists()) {
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
      ...entries.where(
        (entry) =>
            entry['name'] != model.name || entry['version'] != model.version,
      ),
      model.copyWith(isDownloaded: true, isSelected: false).toJson(),
    ];
    await _writeRegistryRaw(updatedEntries);
  }

  Future<void> _removeDownloadedModelFromRegistry(
    AvailableModelPreview model,
  ) async {
    final entries = await _readRegistryRaw();
    final updatedEntries = entries
        .where(
          (entry) =>
              entry['name'] != model.name || entry['version'] != model.version,
        )
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
        merged[key] = existing.copyWith(
          path: localModel.path,
          isDownloaded: true,
          labels: localModel.labels.isNotEmpty ? localModel.labels : existing.labels,
          inputSize: localModel.inputSize,
          confidenceThreshold: localModel.confidenceThreshold,
        );
        continue;
      }

      merged[key] = localModel.copyWith(
        isAvailableInApi: false,
        isDownloaded: true,
        downloadUrl: null,
        isSelected: false,
      );
    }

    return merged.values.toList()..sort(_compareModels);
  }

  String _modelKey(AvailableModelPreview model) {
    return '${model.name}::${model.version}';
  }

  bool _isSameModel(AvailableModelPreview a, AvailableModelPreview b) {
    return a.name == b.name && a.version == b.version;
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

  static const _fallbackModels = [
    AvailableModelPreview(
      name: 'Traffic Sign Detector',
      version: 'v1.0.0',
      path: 'assets/models/traffic_sign_detector.tflite',
      isDownloaded: false,
      isSelected: true,
      labels: [
        'Ograniczenie predkosci',
        'Zakaz wjazdu',
        'Stop',
        'Ustap pierwszenstwa',
        'Przejscie dla pieszych',
        'Nakaz jazdy prosto',
      ],
      inputSize: 320,
      confidenceThreshold: 0.45,
    ),
  ];
}
