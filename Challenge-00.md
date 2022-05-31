# Challenge 0: 前提条件

**[Home](./README.md)** - [Next Challenge >](./Challenge-01.md)

## Introduction

あなたは急ごしらえの AI 需要予測プロジェクトに招集された Adventure Works 社のデータサイエンティストです。まだ発足したばかりなので、データサイエンティストは自分一人しかいません。しかし上層部の要求によりすぐに PoC を始めなければならず、残念ながらハイスペックなマシンを調達したり、AI プロジェクトに必要なツール類を制約の多い業務 PC に導入している時間がありません。ただし幸運なことにある程度の額の Azure サブスクリプションが提供されました。

## Rules
**チームメンバーは Hack セクションに記載されたタスクを行い、成功基準を満たす必要があります。タスクが完了した時点でコーチに成功基準を満たしていることを説明し、合格すれば次のチャレンジに進むことができます。**

**課題を解くためのヒントは学習リソース セクションの中に必ずあります。Hack がスタックしてしまったら学習リソースに立ち返ってください。**


## Description

DevOps と同様、MLOps は非常に幅広いトピックであり、使用するツールに関しても多くの選択肢があります。このチャレンジでは、コンピュータに適切なツールがインストールされていることを確認することに焦点を当てます。

本チャレンジでは、[Azure Machine Learning スタジオ](https://docs.microsoft.com/azure/machine-learning/overview-what-is-machine-learning-studio)と呼ばれるクラウドプラットフォーム上で開発を行います。

![Azure Machine Learning Workspace](./images/001.png)
Azure Machine Learning ワークスペースと各アセットとリソースについて図と[解説](https://docs.microsoft.com/azure/machine-learning/v1/concept-azure-machine-learning-architecture)を見ながら関係を理解します。

## 前提条件
- Azure のサブスクリプションを取得していること。お持ちでない場合は、[無料トライアル](https://azure.microsoft.com/free/)にお申し込みください。以下の Azure リソースを作成できることを確認します。
  - Application Insights
  - Azure Container Registry
  - Azure Container Instance
  - Azure Machine Learning
  - KeyVault
  - Storage Account

- 提供された無償の Azure Pass がある場合は、そのまま [Azure Portal](https://portal.azure.com/) にログインできます。認証時の多要素認証は本 Hack ではスキップできます。
- 本コンテンツは Azure ML SDK v1 向けに作成されています。

## Hack
以下の Azure Machine Learning アセットとリソースを作成します。Azure Machine Learning スタジオ UI から作成しても、Azure Machine Learning SDK から作成してもかまいません。
1. [Azure Machine Learning ワークスペース](https://ml.azure.com/)の作成。これは機械学習モデルの実験、学習、デプロイに使用する、クラウド上の基礎的なリソースです。
1. コンピューティング インスタンスの作成（汎用 Standard_D2s_v3 推奨）
1. コンピューティング クラスターの作成（汎用 Standard_D2s_v3 推奨）

<details><summary>ご自分のマシン上で作業する場合（オプション、今回は対象外）</summary><div>

- Python のインストール、バージョンは最低でも 3.6.5 です。データサイエンス系の作業には Anaconda がより好ましい。
  - Anaconda - <https://docs.anaconda.com/anaconda/install/windows/>
  - Miniconda - <https://docs.conda.io/en/latest/miniconda.html>
  - Python - <https://www.python.org/downloads/>

  **Note:** 自分の OS の環境を壊さずに OpenHack 用の conda 環境を用意するのがおすすめです。インストールができない方は、Azure Machine Learning の Notebook 上で実行してください。

- Python モジュールが pip 経由でダウンロード可能であることを確認する（[PyPI](https://pypi.org)から、または内部のパッケージマネージャから）。
  - arima-env.yml を参照
  - conda env create -f=arima-env.yml
- Visual Studio Code または任意の Python IDE（Jupyter Notebook がよく使われます）
  - Python extensions


</div></details>

<br/>

## 成功基準

- Azure Machine Learning ワークスペースが Azure サブスクリプションに作成されている。
- Azure Machine Learning ワークスペースにコンピューティング インスタンスおよびコンピューティング クラスターが作成されている。
- Python がインストールされており、Python コードを実行するための IDE がある。（オプション、今回は対象外）

## 学習リソース
- [Azure Machine Learning のしくみ:アーキテクチャと概念](https://docs.microsoft.com/azure/machine-learning/v1/concept-azure-machine-learning-architecture)
- [Azure Machine Learning ワークスペースとは](https://docs.microsoft.com/azure/machine-learning/concept-workspace)
- [Azure Machine Learning の利用を開始するために必要なワークスペース リソースを作成する](https://docs.microsoft.com/azure/machine-learning/quickstart-create-resources)
- [Azure Machine Learning コンピューティング インスタンスとは](https://docs.microsoft.com/azure/machine-learning/concept-compute-instance)
- [Azure Machine Learning コンピューティング クラスターとは](https://docs.microsoft.com/azure/machine-learning/how-to-create-attach-compute-cluster)