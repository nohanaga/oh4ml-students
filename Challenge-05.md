# Challenge 5 – 機械学習パイプラインのトリガーとオーケストレーション

[< Previous Challenge](./Challenge-04.md) - **[Home](./README.md)** - [Next Challenge >](./Challenge-06.md)

## Introduction
機械学習パイプラインの構築によってトレーニング・デプロイコードのモジュール化が進み、品質や生産性が向上してきました。すでに機械学習パイプラインができているので、あとは何らかのイベントや状態変化に応じてトレーニングが自動実行されれば非常に効率的です。現状トレーニングに用いるデータは継続的に更新されるため、その都度トレーニングを手動で起動するのは面倒になってきています。

この課題の目的は、機械学習パイプラインをプログラムでスケジュールする方法について理解することです。

## Description
機械学習パイプラインは時刻またはファイルシステムの変更に基づいてスケジュールを作成することができます。時刻ベースのスケジュールを使用すると、データの誤差の監視などの日常的なタスクを行うことができます。 変更ベースのスケジュールを使用すると、新しいデータがアップロードされたり古いデータが編集されたりといった、不規則な変更や予期しない変更に対処できます。

また、機械学習パイプラインを Azure Data Factory などのデータインジェストパイプラインや Azure DevOps、Github Actions などのより大きな MLOps オーケストレーション パイプラインの一部として呼び出すことができます。

## ツールの選択
MLOps オーケストレーションを実現するための[複数のテクノロジーの違い](https://docs.microsoft.com/azure/architecture/example-scenario/mlops/aml-decision-tree#overall-orchestration-and-scheduling)を理解し、チームで取り組むテクノロジーを選択してください。下記のいずれかを満たす必要があります。

1. モデルのトレーニングにフォーカスした自動化の場合、機械学習パイプラインのトリガー機能を使用します。
1. Azure Logic App を使用してより複雑なトリガーを実装します。
1. Github Actions の Workflow を使ってパイプラインを実装しコードコミット駆動を使用します。
1. Azure DevOps の Azure Pipelines を使ってパイプラインを実装しコードコミット駆動を使用します。

## Hack
### 1. 機械学習パイプライン
1. 新しいノートブックを作成します。
1. [Challenge 4](./Challenge-04.md) で作成したパイプラインID を使用して 5 分間隔で起動する時刻ベースのスケジュールを作成して複数回実行されることを確認します。（実際は月次など適切な更新間隔にセットします）
1. 作成した時刻ベースのパイプラインのスケジュールを無効化します。
1. [Challenge 4](./Challenge-04.md) で作成したパイプラインID を使用して特定のフォルダの変更を監視する変更ベースのスケジュールを作成します。
1. 作成した変更ベースのパイプラインのスケジュールを無効化します。

### 2. Azure Logic App
[複雑なトリガーに Azure Logic Apps を使用する](https://docs.microsoft.com/azure/machine-learning/how-to-trigger-published-pipeline#use-azure-logic-apps-for-complex-triggers) を参照して実装します。

### 3. Github Actions
[Azure Machine Learning で GitHub Actions を使用する](https://docs.microsoft.com/azure/machine-learning/how-to-github-actions-machine-learning?view=azure-devops) を参照して実装します。ワークフローの構築に使用するアセットは、[こちら](https://github.com/tsmatsuz/oh4ml-lite-github-workflow)のリポジトリを使用してください。

  **注意:** すでに**有効な GitHub アカウントを所有している方**のみ選択してください。

### 4. Azure DevOps
[Azure Pipelines を使用してデータの準備、機械学習モデルのトレーニング、デプロイ、監視を行う](https://docs.microsoft.com/azure/devops/pipelines/targets/azure-machine-learning?context=azure%2Fmachine-learning%2Fcontext%2Fml-context&view=azure-devops&tabs=classic) を参照して実装します。Azure Pipelines を使えば今回作成した機械学習パイプラインと同等のものを実装できますし、コードコミット駆動やリリース ゲートと承認機能を追加することができます。経験のあるエンジニアは Azure Pipelines でトレーニング・デプロイパイプラインを構築しても構いませんが、今回は機械学習パイプラインの起動のみにフォーカスします。

  **注意:** 無料アカウントで Azure DevOps を使用している場合パイプライン実行時に Requiest Form があり申請に 2,3 日かかります。**すでにパイプラインが実行可能な方のみ**選択してください。


## 成功基準
- 選択したテクノロジーを使用して任意のトリガーによってパイプラインを起動できること。

## 学習リソース
 - [最適なツール選択のための Azure Machine Learning 意思決定ガイド](https://docs.microsoft.com/azure/architecture/example-scenario/mlops/aml-decision-tree)
 - [機械学習パイプラインをトリガーする](https://docs.microsoft.com/azure/machine-learning/how-to-trigger-published-pipeline)
 
 