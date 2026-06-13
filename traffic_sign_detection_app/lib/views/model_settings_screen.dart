import 'package:flutter/material.dart';

import '../models/available_model_preview.dart';
import '../service/model_service.dart';
import '../widgets/model_card.dart';

class ModelSettingsScreen extends StatefulWidget {
  const ModelSettingsScreen({super.key});

  @override
  State<ModelSettingsScreen> createState() => _ModelSettingsScreenState();
}

class _ModelSettingsScreenState extends State<ModelSettingsScreen> {
  final ModelService _modelService = ModelService.instance;
  bool _isFetchingModels = false;

  @override
  void initState() {
    super.initState();
    _retryFetchModels();
  }

  Future<void> _retryFetchModels() async {
    setState(() {
      _isFetchingModels = true;
    });

    await _modelService.fetchAvailableModels(ModelService.defaultModelsApiUri);

    if (mounted) {
      setState(() {
        _isFetchingModels = false;
      });
    }
  }

  Future<void> _selectModel(AvailableModelPreview selectedModel) async {
    await _runModelAction(() => _modelService.loadModel(selectedModel));
  }

  Future<void> _downloadModel(AvailableModelPreview model) async {
    await _runModelAction(() => _modelService.downloadModel(model));
  }

  Future<void> _updateModel(AvailableModelPreview model) async {
    await _runModelAction(() async {
      await _modelService.downloadModel(model);
      final downloadedModel = _modelService.availableModels.firstWhere(
        (candidate) =>
            candidate.name == model.name && candidate.version == model.version,
        orElse: () => model,
      );
      if (downloadedModel.isDownloaded) {
        await _modelService.loadModel(downloadedModel);
      }
    });
  }

  Future<void> _deleteModel(AvailableModelPreview model) async {
    await _runModelAction(() => _modelService.deleteModel(model));
  }

  Future<void> _runModelAction(Future<void> Function() action) async {
    await action();
    if (!mounted || _modelService.lastError == null) {
      return;
    }

    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text(_modelService.lastError!)));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Ustawienia modeli')),
      body: AnimatedBuilder(
        animation: _modelService,
        builder: (context, _) {
          final models = _modelService.availableModels;
          final fetchError = _modelService.lastError;

          return Stack(
            children: [
              Column(
                children: [
                  if (fetchError != null)
                    MaterialBanner(
                      content: Text(fetchError),
                      actions: [
                        TextButton(
                          onPressed:
                              _isFetchingModels ? null : _retryFetchModels,
                          child: const Text('Spróbuj ponownie'),
                        ),
                      ],
                    ),
                  Expanded(
                    child: models.isEmpty
                        ? Center(
                            child: Padding(
                              padding: const EdgeInsets.all(24),
                              child: Text(
                                fetchError == null
                                    ? 'Brak dostępnych modeli.'
                                    : 'Brak modeli z API.\n'
                                        'Adres: ${ModelService.defaultModelsApiUri}',
                                textAlign: TextAlign.center,
                              ),
                            ),
                          )
                        : ListView.separated(
                            padding: const EdgeInsets.all(16),
                            itemCount: models.length,
                            separatorBuilder: (_, _) =>
                                const SizedBox(height: 12),
                            itemBuilder: (context, index) {
                              final model = models[index];

                              return ModelCard(
                                model: model,
                                onUse: () => _selectModel(model),
                                onDownload: () => _downloadModel(model),
                                onUpdate: () => _updateModel(model),
                                onDelete: () => _deleteModel(model),
                              );
                            },
                          ),
                  ),
                ],
              ),
              if (_isFetchingModels || _modelService.isBusy)
                const Positioned.fill(
                  child: ColoredBox(
                    color: Color(0x88000000),
                    child: Center(child: CircularProgressIndicator()),
                  ),
                ),
            ],
          );
        },
      ),
    );
  }
}
