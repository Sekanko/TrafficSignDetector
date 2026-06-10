import 'package:flutter/material.dart';

import '../models/available_model_preview.dart';
import '../widgets/model_card.dart';

class ModelSettingsScreen extends StatefulWidget {
  const ModelSettingsScreen({super.key});

  @override
  State<ModelSettingsScreen> createState() => _ModelSettingsScreenState();
}

class _ModelSettingsScreenState extends State<ModelSettingsScreen> {
  // TODO: Replace mock data with ModelService.
  final List<AvailableModelPreview> _models = [
    AvailableModelPreview(
      name: 'Linear',
      version: 'v1.0.0',
      path: 'models/linear_v1.ptl',
      isDownloaded: true,
      isSelected: false,
    ),
    AvailableModelPreview(
      name: 'CNN',
      version: 'v2.1.0',
      path: 'models/cnn_v2.ptl',
      isDownloaded: true,
      isSelected: true,
    ),
    AvailableModelPreview(
      name: 'MobileNet',
      version: 'v1.4.2',
      path: 'models/mobilenet_v1.ptl',
      isDownloaded: false,
      isSelected: false,
    ),
  ];

  void _selectModel(AvailableModelPreview selectedModel) {
    setState(() {
      for (var index = 0; index < _models.length; index++) {
        final model = _models[index];
        _models[index] = model.copyWith(
          isSelected: model.path == selectedModel.path,
        );
      }
    });
  }

  void _downloadModel(AvailableModelPreview model) {
    // TODO: Download model with ModelService.
  }

  void _updateModel(AvailableModelPreview model) {
    // TODO: Update local model with ModelService.
  }

  void _deleteModel(AvailableModelPreview model) {
    // TODO: Delete local model with ModelService.
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Ustawienia modeli')),
      body: ListView.separated(
        padding: const EdgeInsets.all(16),
        itemCount: _models.length,
        separatorBuilder: (_, __) => const SizedBox(height: 12),
        itemBuilder: (context, index) {
          final model = _models[index];

          return ModelCard(
            model: model,
            onUse: () => _selectModel(model),
            onDownload: () => _downloadModel(model),
            onUpdate: () => _updateModel(model),
            onDelete: () => _deleteModel(model),
          );
        },
      ),
    );
  }
}
