# Challenge 1 – ローカルでモデルを作成する

[< Previous Challenge](./Challenge-00.md) - **[Home](./README.md)** - [Next Challenge >](./Challenge-02.md)

## Introduction

この課題の目的は、時系列 [ARIMA](https://ja.wikipedia.org/wiki/%E8%87%AA%E5%B7%B1%E5%9B%9E%E5%B8%B0%E5%92%8C%E5%88%86%E7%A7%BB%E5%8B%95%E5%B9%B3%E5%9D%87%E3%83%A2%E3%83%87%E3%83%AB) モデリングを使って日々の取引を予測することです。この課題には、[AdventureWorks](https://docs.microsoft.com/previous-versions/sql/sql-server-2008/ms124501(v=sql.100)) データベースからのトランザクションデータの処理、分析、ARIMA モデルの作成と登録、そして最後に Azure Container Instances(ACI) へのモデルのデプロイが含まれます。このライフサイクル全体は、[Azure Machine Learning Python SDK v1](https://docs.microsoft.com/python/api/overview/azure/ml/?view=azure-ml-py) を使用して行われます。

時系列とは、一定の時間ごとに時間順に収集またはインデックス化された一連のデータポイントのことです。等間隔に連続した時点で取得された一連のデータです。例えば、1時間ごとに収集された気象データ、1日ごとに収集された株価データ、1分ごとに収集されたセンサーデータなどが挙げられます。その結果、時系列データセットには傾向や季節性が見られます。このような時間的な性質が、他の従来のデータセットとは異なり、異なるタイプのモデリングを保証しているのです。予測は機械学習で最も一般的なタスクの 1 つであるため、このチャレンジではこれを体験します。

今回は、事前にデータエンジニアによってデータウェアハウス内に蓄積されたトランザクションデータを利用する想定とします。

## Description
多くのデータ サイエンスや機械学習の実験は、ノートブック でコードを実行することによって行われます。Azure Machine Learning の**コンピューティング インスタンス**には、大規模な作業に使用できる完全な機能を備えた Python ノートブック環境 (Jupyter と JuypyterLab) が含まれています。基本的なノートブックの編集には、Azure Machine Learning スタジオに組み込まれている ノートブック ページを使用してチャレンジを行います。今回はこのノートブック実行環境を**ローカル環境**と呼びます。

## Hack
1. Azure Machine Learning スタジオのホームから **Notebooks** の「今すぐ開始」をクリックし、コーチが提供する Git リポジトリ `https://github.com/nohanaga/oh4ml-students.git` をワークスペースにクローンします。
1. ノートブックのマイ ファイル ウィンドウで ↻ をクリックしてビューを更新し、新しい /Users/\<your-user-name\>/oh4ml-students フォルダーが作成されていることを確認します。
1. /Notebooks フォルダ内の `00_LoadData.ipynb` を開き、[Challenge 0](./Challenge-00.md) で作成した**コンピューティング インスタンス**を割り当て、カーネルを「Python 3.8 - AzureML」にセットしセルを上から実行してデータをダウンロードしてデータセットとしてインポートします。

    **注意:** ノートブック上部に *"いずれかの Azure SDK を使用するには、コンピューティングに対して認証される必要があります。認証されるには、認証ボタンを使用してください。"* と表示された場合は認証ボタンを押してアカウントの認証を行います。

    **Tips:** Jupyter Notebook や JupyterLab の UI を使用したい方は、**コンピューティング インスタンス**のリストから起動することができます。


1. `01_TrainOnLocal.ipynb` を開き、セルを上から実行して ARIMA トレーニングの一連の処理を体験します。最後のセルでは、作成したモデルによる予測結果と精度を出力します。

  **注意:** これはデータサイエンスの領域ですので詳細なモデル開発手法については扱いません。このハックの焦点はデータサイエンスではなく、**MLOps** にあり、ML プロジェクトを加速し、ML ワークフローの効率、品質、一貫性を高めるために、DevOps のプラクティスと原則をどのように適用できるかを理解することにあります。

## 成功基準

- 学習に使うトランザクションデータが年単位に分割された状態でデータセットに登録されている。
- ローカル環境で ARIMA モデルを作成し予測結果と精度を出力する。モデルの精度の良し悪しは問いません。


## 学習リソース
 - [ワークスペース内のファイルを作成および管理する方法](https://docs.microsoft.com/azure/machine-learning/how-to-manage-files)
 - [Azure Machine Learning ワークスペースを作成して探索する](https://microsoftlearning.github.io/DP-100JA-Designing-and-Implementing-a-Data-Science-Solution-on-Azure/Instructions/01-create-a-workspace.html)
 - [ワークスペースで Jupyter Notebook を実行する](https://docs.microsoft.com/azure/machine-learning/how-to-run-jupyter-notebooks)