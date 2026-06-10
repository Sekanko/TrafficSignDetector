import 'package:flutter/material.dart';

import '../models/available_model_preview.dart';

class ModelCard extends StatelessWidget {
  const ModelCard({
    required this.model,
    required this.onUse,
    required this.onDownload,
    required this.onUpdate,
    required this.onDelete,
    super.key,
  });

  final AvailableModelPreview model;
  final VoidCallback onUse;
  final VoidCallback onDownload;
  final VoidCallback onUpdate;
  final VoidCallback onDelete;

  @override
  Widget build(BuildContext context) {
    return Card(
      color: const Color(0xFF151A21),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    model.name,
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.w700,
                        ),
                  ),
                ),
                Text(
                  model.version,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: Theme.of(context).colorScheme.onSurfaceVariant,
                      ),
                ),
              ],
            ),
            const SizedBox(height: 14),
            if (!model.isDownloaded)
              Align(
                alignment: Alignment.centerRight,
                child: FilledButton.tonal(
                  onPressed: onDownload,
                  child: const Text('Pobierz'),
                ),
              )
            else
              Row(
                children: [
                  Expanded(
                    child: FilledButton(
                      onPressed: model.isSelected ? null : onUse,
                      child: _ButtonText(model.isSelected ? 'Aktywny' : 'Użyj'),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: OutlinedButton(
                      onPressed: onUpdate,
                      child: const _ButtonText('Zaktualizuj'),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: TextButton(
                      onPressed: onDelete,
                      child: const _ButtonText('Usuń'),
                    ),
                  ),
                ],
              ),
          ],
        ),
      ),
    );
  }
}

class _ButtonText extends StatelessWidget {
  const _ButtonText(this.text);

  final String text;

  @override
  Widget build(BuildContext context) {
    return FittedBox(
      fit: BoxFit.scaleDown,
      child: Text(
        text,
        maxLines: 1,
      ),
    );
  }
}
