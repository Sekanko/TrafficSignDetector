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
    final canDownload = model.canDownloadFromApi;
    final canUpdate = model.hasUpdate;
    final statusText = canUpdate
        ? 'Zaktualizuj'
        : model.isAvailableInApi
        ? 'Aktualny'
        : 'Tylko lokalny';

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
                if (!model.isAvailableInApi && model.isDownloaded)
                  Padding(
                    padding: const EdgeInsets.only(right: 8),
                    child: Text(
                      'Lokalny',
                      style: Theme.of(context).textTheme.labelSmall?.copyWith(
                        color: Theme.of(context).colorScheme.primary,
                      ),
                    ),
                  ),
                if (model.hasUpdate)
                  Padding(
                    padding: const EdgeInsets.only(right: 8),
                    child: Text(
                      'Aktualizacja',
                      style: Theme.of(context).textTheme.labelSmall?.copyWith(
                        color: Theme.of(context).colorScheme.tertiary,
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
            const SizedBox(height: 8),
            Text(
              model.isPipeline
                  ? 'Detektor znaków drogowych'
                  : 'Pojedynczy model detekcji',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 14),
            if (!model.isDownloaded)
              Align(
                alignment: Alignment.centerRight,
                child: FilledButton.tonal(
                  onPressed: canDownload ? onDownload : null,
                  child: Text(canDownload ? 'Pobierz' : 'Brak adresu'),
                ),
              )
            else
              Row(
                children: [
                  Expanded(
                    child: FilledButton(
                      onPressed: model.isSelected ? null : onUse,
                      style: model.isSelected
                          ? FilledButton.styleFrom(
                              disabledBackgroundColor: Colors.green.shade700,
                              disabledForegroundColor: Colors.white,
                            )
                          : null,
                      child: _ButtonText(model.isSelected ? 'Aktywny' : 'Użyj'),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: OutlinedButton(
                      onPressed: canUpdate ? onUpdate : null,
                      style: canUpdate
                          ? OutlinedButton.styleFrom(
                              backgroundColor: Colors.amber.shade600,
                              foregroundColor: Colors.black,
                              side: BorderSide(color: Colors.amber.shade600),
                            )
                          : null,
                      child: _ButtonText(statusText),
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
    return FittedBox(fit: BoxFit.scaleDown, child: Text(text, maxLines: 1));
  }
}
