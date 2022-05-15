# Challenge 4 – トレーニング・デプロイパイプラインの作成

[< Previous Challenge](./Challenge-03.md) - **[Home](./README.md)** - [Next Challenge >](./Challenge-05.md)

## Introduction
目標の精度を達成した予測モデルと魅力的なビジュアライズの効果もあり PoC プロジェクトは上層部の支持を得ることができ、定常的な社内プロジェクトとして昇格しました。今後は技術の横展開や他のシステムのデータに対して同様の予測を行うことが求められています。それに伴いデータサイエンティストの作業量は増大しており、より高度な並列化・自動化が必要です。現状は各々のデータサイエンティストがノートブック上でモデル作成のためのコードを記述しており、中には車輪の再発明的な無駄な作業が生じたり、モデル作成に必要な前処理の一つが抜けてしまったり、うまく共同作業ができなかったりしています。さらに情報システム担当役員からモデルの品質管理の強化に関する通達が出され、品質管理部門の監査に対してレポートする際の実験証跡も管理する必要があります。

この課題の目的は、**機械学習パイプライン**を使用して、機械学習ワークフローの構築、最適化、管理を行う方法について理解することです。

## Description

**機械学習パイプライン**を使用し、機械学習フェーズをつなげるワークフローを作成して管理します。たとえば、パイプラインには、データ準備、モデル トレーニング、モデル デプロイ、推論/スコアリングの各フェーズが含まれることが考えられます。それぞれのフェーズには、複数のステップを含めることができ、各ステップは、さまざまなコンピューティング先において無人実行することができます。

![aml-pipelines-concept](./images/004.png)

機械学習の各フェーズをパイプライン化することのメリットを[解説](https://docs.microsoft.com/azure/machine-learning/concept-ml-pipelines#key-advantages)を参照しながら理解します。

Azure Machine Learning のパイプラインの複数のステップを組み合わせて構成して、共有可能で再利用可能な Azure Machine Learning ワークフローであるパイプラインを構築できます。 パイプラインの各ステップは、ステップの内容 (スクリプトと依存関係) に加えて入力とパラメーターが変更されていない場合に、以前の実行結果を再利用できるように構成することができます。

Azure Machine Learning SDK を介して使用できる組み込みステップは多数あります。[azureml.pipeline.steps パッケージ](https://docs.microsoft.com/python/api/azureml-pipeline-steps/azureml.pipeline.steps?view=azure-ml-py)のリファレンス ドキュメントを参照してください。本チャレンジで用いるのは、最も柔軟性の高いクラスである Python スクリプトを実行する `PythonScriptStep` です。これは Python コードを自由に記述できるため、上図のような機械学習フェーズを自分の好みの粒度でステップ化できます。今回はここから 3 つのフェーズをステップとして切り出してパイプライン化を試みます。

## Hack

1. 新しいノートブックを作成します。
1. パイプラインの手順で必要なデータへのアクセスに使用されるデータストアを設定します。
1. パイプラインの手順間で一時データを渡すために共通の中間データセット `OutputFileDatasetConfig` オブジェクトを構成します。
1. トレーニングの実行環境を `RunConfiguration` オブジェクトを作成して構成します。
1. `train.py` を使用してトレーニングを実行する 1 つ目のパイプライン ステップを作成します。
1. `register.py` を作成してモデルを登録する 2 つ目のパイプライン ステップを作成します。
1. `deploy.py` を作成してモデルを `Azure Container Instances(ACI)` にデプロイする 3 つ目のパイプライン ステップを作成します。
1. パイプライン ステップからパイプラインを構成します。
1. 実験でパイプラインを実行します。
1. パイプラインを発行します。
1. 発行されたパイプラインを実行します。

## 成功基準
- 作成したパイプラインのステップが `[train, register, deploy]` の 3 ステップで構成されていること。
- パイプラインエンドポイントにパイプラインがアクティブ状態で登録されていること。
- 発行されたパイプラインが正常に実行されること。


<details>
<summary>ヒント</summary>

- パイプライン ステップは 1 つずつ作成し、正常実行を確認しながらつなげていくことをおすすめします。
- パイプライン ステップの実行順序は、`OutputFileDatasetConfig` オブジェクトを入力として用いるか出力として用いるかの違いによって[暗黙的に決定](https://docs.microsoft.com/azure/machine-learning/how-to-move-data-in-out-of-pipelines#use-outputfiledatasetconfig-for-intermediate-data)されます。
- パイプライン ステップの実行順序を明示的に指定する場合は [run_after](https://docs.microsoft.com/python/api/azureml-pipeline-core/azureml.pipeline.core.pipelinestep?view=azure-ml-py#azureml-pipeline-core-pipelinestep-run-after) メソッドを使用します。
- 外部からパイプラインを実行する際に[パイプライン パラメータ](https://docs.microsoft.com/azure/machine-learning/how-to-create-machine-learning-pipelines#use-pipeline-parameters-for-arguments-that-change-at-inference-time)を指定すると各パイプライン ステップ内のスクリプトへ引数を渡すことができます。実行時の指定のしかたは[こちら](https://docs.microsoft.com/azure/machine-learning/how-to-deploy-pipelines#run-a-published-pipeline)を参照。


</details>
<br>


## 学習リソース
 - [Azure Machine Learning パイプラインとは](https://docs.microsoft.com/azure/machine-learning/concept-ml-pipelines)
 - [Why build pipelines?](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/machine-learning-pipelines#why-build-pipelines)
 - [Azure Machine Learning SDK で機械学習パイプラインを作成して管理する](https://docs.microsoft.com/azure/machine-learning/how-to-create-machine-learning-pipelines)
 - [機械学習パイプラインを発行して追跡する](https://docs.microsoft.com/azure/machine-learning/how-to-deploy-pipelines)

## さらなる学習
 - トレーニング・デプロイパイプラインは、Azure DevOps や Github Actions を使っても構築することができます。Azure DevOps/Github Actions を使って本チャレンジのパイプラインを構築してください。

    - [Azure Pipelinesを使用してデータの準備、機械学習モデルのトレーニング、デプロイ、監視を行う](https://docs.microsoft.com/azure/devops/pipelines/targets/azure-machine-learning?context=azure%2Fmachine-learning%2Fcontext%2Fml-context&view=azure-devops&tabs=yaml)
    - [Azure Machine Learning で GitHub Actions を使用する](https://docs.microsoft.com/azure/machine-learning/how-to-github-actions-machine-learning?view=azure-devops)

    **注意:** 無料アカウントで Azure DevOps を使用している場合パイプライン実行時に Requiest Form があり申請に 2,3 日かかります。**すでにパイプラインが実行可能な方のみ**選択してください。

 - データ加工フェーズをパイプライン化することも重要です。`00_LoadData.ipynb` の各セルを分離し、データ変換ステップを追加してパイプラインを構築してください。