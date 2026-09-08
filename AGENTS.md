# Regras do projeto

## Protocolo obrigatório de Machine Learning

Leia e siga `.docs/regra-metodologica-ml.md` antes de revisar, alterar ou executar qualquer experimento.

Pontos inegociáveis:

- Target deve manter codificação documentada: `0 = 0–1 consulta`, `1 = 2+ consultas`; validar codebook e exibir distribuição antes de treino.
- Usar `ParameterGrid` manual com `StratifiedKFold` externo e interno. Não trocar por `GridSearchCV` sem aprovação explícita.
- Nenhuma transformação adaptativa pode usar validação ou teste: imputação, encoding, scaler, MDI, resampling, pesos, calibração, hiperparâmetros, ensemble e limiar ajustam somente no treino interno.
- `x_test` e `y_test` externos servem somente para avaliação final.
- AUROC e AP/PR-AUC usam probabilidades ou decision scores; nunca resultado de `predict()`.
- Avaliar sempre baseline `DummyClassifier`, métricas por classe, macro F1, balanced accuracy, MCC, matriz `labels=[0, 1]` e predições OOF.
- Baseline com todas features é obrigatório. MDI 80%, custo sensível, ensemble e limiar são cenários separados.
- Não usar SMOTE: dados sensíveis não devem gerar amostras sintéticas. NearMiss não é padrão e exige aprovação explícita.
- Não tratar importance preditiva como causalidade ou recomendação clínica.
- Não alterar README, artigo ou interpretação clínica automaticamente; usar marcação de revisão manual quando faltar evidência.

## Organização atual

- Pipeline experimental: `src/AUSMI.ipynb`.
- Dados: `src/NPHA-doctor-visits.csv`.
- Plano ativo: `.docs/plano-revisao-experimental.md`.
- Estado de coordenação: `.slim/deepwork/revisao-experimentos-ml.md`.

## Fluxo de trabalho

1. Inspecionar estado atual e preservar resultados legados como não comparáveis.
2. Apresentar diagnóstico e plano antes de mudar experimento.
3. Pedir aprovação do usuário antes de implementar.
4. Validar notebook do início ao fim e salvar artefatos reproduzíveis.
5. Reportar limites: n=714, desenho transversal, base única e sem validação externa.
