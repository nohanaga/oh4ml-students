# Challenge 7 – CLI v2+YAML による MLOps（ストレッチ課題）

[< Previous Challenge](./Challenge-06.md) - **[Home](./README.md)**

## Introduction

Azure Machine Learning CLI v2 は、Azure Machine Learning の拡張インターフェイスです。これにより、**コマンド ライン** からモデルをトレーニングおよびデプロイできます。また、モデルのライフサイクルを追跡しながらデータ サイエンスのスケールアップとスケールアウトを加速する機能もあります。

これまで、コマンドラインインターフェイス(CLI)を使用したクラウドリソースの管理に慣れ親しんでいるエンジニアは、その操作感をそのまま機械学習に持ち込むことができます。

## Description
本チャレンジでは　これまで作成した Python コードの一部を CLI v2 を使用して書き換えて実行します。Azure Machine Learning で使えるコマンドの一覧は、[CLI v2 リファレンス](https://docs.microsoft.com/cli/azure/ml?view=azure-cli-latest)を参照してください。

## Hack
1. [Challenge 0](./Challenge-00.md) で使用した `00_LoadData.ipynb` を開き、最終的に作成した 3 つの CSV ファイルを CLI v2 を使用してアップロードしデータセットとして登録します。
1. [Challenge 2](./Challenge-02.md) で作成したノートブックを開き、 CLI v2 を使用して学習実行と生成ファイルをローカル環境にダウンロードします。必要に応じて、環境とコンピュート クラスターも作成します。
1. [Challenge 3](./Challenge-03.md) で作成したノートブックを開き、作成したモデルを CLI v2 を使用して Azure Machine Learning ワークスペースに登録します。
1. CLI v2 を使用してオンライン エンドポイントを作成してモデルをデプロイおよびスコアリングします。
1. [Challenge 4](./Challenge-04.md) で作成したノートブックを開き、CLI v2 を使用して
`[train, register, deploy]` の 3 ステップのパイプラインを作成して実行します。

## 成功基準
- 学習に使うトランザクションデータが年単位に分割された状態でデータセットに登録されていること。
- Azure Machine Learning ワークスペースにモデルが登録されていること。
- リアルタイムエンドポイントのデプロイに成功(サービスの状態が Healthy)すること。
- 作成したパイプラインが正常に実行されること。

## 学習リソース

- [CLI v2 のインストールと設定](https://docs.microsoft.com/azure/machine-learning/how-to-configure-cli?tabs=public)
- [CLI v2 リファレンス](https://docs.microsoft.com/cli/azure/ml?view=azure-cli-latest)
- [CLI v2 YAML スキーマ](https://docs.microsoft.com/azure/machine-learning/reference-yaml-overview)
- [オンライン エンドポイントを使用して機械学習モデルをデプロイおよびスコアリングする](https://docs.microsoft.com/azure/machine-learning/how-to-deploy-managed-online-endpoints)
- [Azure Machine Learning CLI でコンポーネントを使用して機械学習パイプラインを作成して実行する](https://docs.microsoft.com/azure/machine-learning/how-to-create-component-pipelines-cli)