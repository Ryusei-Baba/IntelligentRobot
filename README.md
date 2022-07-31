# IntelligentRobot
知能ロボットコンテストに向けた、TensorFlow Liteを用いた深層学習による画像処理です。   
データセットを自作し、以下の3つを認識します。

・red-ball   
・blue-ball   
・yellow-ball

※動作が重いので知能ロボットコンテストで実用するのは難しいと思います。あくまで深層学習を体験するためのものだと考えください。
***
# デモ動画
### [ボール検出](https://youtu.be/1wTWfVXPn9M)
***
# 使用機器
・Raspberry Pi 4 Model B   
・Raspberry Pi Camera V2   
・Coral USB Accelerator   
***
# チュートリアル
## 環境構築   
※Ubuntu MATE 20.04 LTSをラズパイにインストールしましたが、ここでは省略します。（他のサイトをご覧ください）   
まずこの[サイト](https://coral.ai/docs/accelerator/get-started/#requirements)を参考に環境構築を進めてください。ここでは以下のエラーが考えられます。


・[tensorflowのインストール](https://temcee.hatenablog.com/entry/tensorflow_install_error)   
```
$ pip3 install --upgrade --ignore-installed https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.8.0-py3-none-any.whl
```

・有効なopenpgpデータが見つかりません。   
→参考にしたサイトが見れなくなり、どのように解決したか忘れてしまいました。「curlで「自己署名証明書」を受け入れるには」等で検索してみてください。

・updates—   
→これはラズパイの時刻調整で解決しました。

・[無効なファイル名拡張子を持っているため、無視します。](http://ja.uwenku.com/question/p-tueaecyz-gy.html)   
→/etc/apt/sources.list.d内の不必要なファイル（.list以外）を削除（rm -rf ファイル名）してください。

・ロックできません   
→参考にしたサイトが見れなくなり、どのように解決したか忘れてしまいました。「Ubuntu　ロックを取得できない」等で検索してみてください。   
***
## 物体検出
googleが提供しているデータセットで物体検出をしてみましょう。参考にしたサイトは[こちら](https://qiita.com/rhene/items/1d6c267b1b739f337a75)   
### [物体検出動画](https://youtu.be/tYLJrUAgkyY)   

opencvで検出を行う場合は[こちら](https://github.com/Ryusei-Baba/IntelligentRobot/blob/main/ball_detection.py)を参考にすると良いです。物体中心座標を表示できるように[サンプルコード](https://github.com/google-coral/examples-camera/blob/master/opencv/detect.py)を少し変えています。   

考えられるエラー   
・opencv エラー   
→バージョン4→バージョン3   
・coralエラー   
→coralが熱くなっているか確認、冷たかったら抜き差しで解消

### [物体検出動画座標入り](https://youtu.be/rCMAQIY7D6E)
***
# データ収集
チュートリアルが終わったら、データセットを作ってみましょう。   

まず、データを集めます。
### [データ収集動画](https://youtu.be/IRnpN5iPdIw)   
深層学習での認識率はこのデータセットに依存するので、以下のことに気を付けます。（今回は考慮していません）   
・照明の影響   
・オクルージョンの影響   
・カメラの画角   
・その他   
***
# アノテーション
アノテーション（教師データの作成）を行います。今回使用したツールは[IBM Cloud Annotations](https://www.ibm.com/jp-ja/cloud)です。アカウント作成時にクレジットカード番号等聞かれますが、基本無料ですので入力してください。（※2022年6月現在は無料でしたが、この先変更があるかもしれません。その場合無料でない可能性もあるのでよく確認してください。私の場合、ログイン画面で「200 米ドルのクレジットで開始する」と出てきましたが、こちらの[サイト](https://www.ibm.com/jp-ja/cloud/free#2891977)に詳しく書いてあったため、安心してアカウントを作成することができました。）

ログイン後はこちらの[サイト](https://dream-soft.mydns.jp/blog/developper/smarthome/2021/02/2809/)を参考にアノテーションを進めてください。

※データ収集をiPhoneで行った場合、それはMOV形式の可能性が高いです。IBM Cloud Annotationsではmp4形式でないと扱えないので、動画の変換を行う必要があります。私の場合は[Adobeの無料オンラインツール](https://express.adobe.com/ja-JP/sp/tools/convert-to-mp4)を用いてmp4に変換を行いました。

**アノテーション**
![24fababc488dac3188955652f62248f7](https://user-images.githubusercontent.com/92899820/181877101-a9528cb2-43d9-4871-8ec8-1f76cf1910de.png)

**アクセスキー**   
テキストファイルなどに保存してください
```
credentials = {
  "bucket": "ir-ball",
  "access_key_id": "d98b031274a9498ea7448eb1b4ac0454",
  "secret_access_key": "0820bdaf69065f94899137b24fb422eb8397a3e942dd3dd0",
  "endpoint_url": "https://s3.us.cloud-object-storage.appdomain.cloud"
}
```
***
# モデルの学習
モデルの学習はこちらの[サイト](https://dream-soft.mydns.jp/blog/developper/smarthome/2021/02/2822/)を参考にさせていただきました。

![419e27034e49246c19133ccce033e53b](https://user-images.githubusercontent.com/92899820/181877709-fe30f8dc-e1ae-4d45-9b87-c98d9967946c.jpg)

**メモ**   
GPUは使用しなくてもいいと思います。

**エラー**   
1.1インストールと初期設定   
→依存関係の修正
```
# %%capture
%tensorflow_version 1.x
!pip uninstall numpy -y
!pip install numpy==1.17.5;
#!pip uninstall tensorflow -y
#!pip install tensorflow;
!pip uninstall lucid -y
!pip uninstall folium -y
!pip install folium==0.2.1;
!pip uninstall imgaug -y
!pip install imgaug;
!pip uninstall albumentations -y
!pip install albumentations;
#実行後にランタイムの再起動が必要「RESTART RUNTIME」ボタンをクリック
```

3.1学習の開始   
ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject
NameError: name 'contrib_training' is not defined   
→pycocotools 2.0.4をpycocotools 2.0.0にすることで解決しました。（こちらの[サイト](https://stackoverflow.com/questions/66060487/valueerror-numpy-ndarray-size-changed-may-indicate-binary-incompatibility-exp)を参考）
```
!pip uninstall pycocotools -y
!pip install pycocotools --no-binary pycocotools
!pip install -q pycocotools==2.0.0
```
***
# Raspberry Piでの動作
**作成した学習モデルをダウンロード**   
1.ラズパイ（環境構築済み）を起動し、ウェブサイト(Firefoxなど)から自身のGoogle driveにログインする。   
2.学習モデル（私の場合はmyModel.zip）をダウンロードし、ubuntuのディレクトリ（作業スペース）にファイルを移動させる。   
その後はこちらの[サイト](https://dream-soft.mydns.jp/blog/developper/smarthome/2021/02/2881/)とこちらの[サイト](https://dream-soft.mydns.jp/blog/developper/smarthome/2021/03/2901/)を参考にしました。   

**メモ**   
VNCにログインしなくても動作しました。

### [動作動画](https://youtu.be/1wTWfVXPn9M)
***
# 参考
[【Raspberry Pi】ディープラーニングで検出したオブジェクトの位置情報を取得する](https://murasan-net.com/index.php/2022/02/12/post-613/)   
[【OpenCV】cv2.rectangle関数の使い方【長方形を描画する】](https://shikaku-mafia.com/opencv-rectangle/)   
[【Python】OpenCVでの図形や文字列の描画まとめ（四角形・線分・矢印・円・楕円・マーク・多角形）](https://qiita.com/atchicken/items/4b17d5b1b8ef014a8331)   
