# IntelligentRobot
知能ロボットコンテストに向けた、TensorFlow Liteを用いた深層学習による画像処理です。   
データセットを自作し、以下の3つを認識します。

・red-ball   
・blue-ball   
・yellow-ball
###
# デモ動画
・[ボール検出](https://youtu.be/1wTWfVXPn9M)
###
# 使用機器
・Raspberry Pi 4 Model B   
・Raspberry Pi Camera V2   
・Coral USB Accelerator   
###
# チュートリアル
**〇環境構築**   
※Ubuntu MATE 20.04 LTSをラズパイにインストールしましたが、ここでは省略します。（他のサイトをご覧ください）   
まずこの[サイト](https://coral.ai/docs/accelerator/get-started/#requirements)を参考に環境構築を進めてください。ここでは以下のエラーが考えられます。


・[tensorflowのインストール](https://temcee.hatenablog.com/entry/tensorflow_install_error)   
→pip3 install --upgrade --ignore-installed https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.8.0-py3-none-any.whl

・有効なopenpgpデータが見つかりません。   
→参考にしたサイトが見れなくなり、どのように解決したか忘れてしまいました。「curlで「自己署名証明書」を受け入れるには」等で検索してみてください。

・updates—   
→これはラズパイの時刻調整で解決しました。

・[無効なファイル名拡張子を持っているため、無視します。](http://ja.uwenku.com/question/p-tueaecyz-gy.html)   
→/etc/apt/sources.list.d内の不必要なファイル（.list以外）を削除（rm -rf ファイル名）してください。

・ロックできません   
→参考にしたサイトが見れなくなり、どのように解決したか忘れてしまいました。「Ubuntu　ロックを取得できない」等で検索してみてください。   


**〇物体検出**
googleが提供しているデータセットで物体検出をしてみましょう。参考にしたサイトは[こちら](https://qiita.com/rhene/items/1d6c267b1b739f337a75)   
[物体検出動画](https://youtu.be/tYLJrUAgkyY)   

opencvで検出を行う場合は[こちら](https://github.com/Ryusei-Baba/IntelligentRobot/blob/main/ball_detection.py)を参考にすると良いです。中心座標を表示できるように[サンプルコード](https://github.com/google-coral/examples-camera/blob/master/opencv/detect.py)を少し変えています。   

考えられるエラー   
・opencv エラー   
→バージョン4→バージョン3   
・coralエラー   
→coralが熱くなっているか確認、冷たかったら抜き差しで解消

[物体検出動画座標入り](https://youtu.be/rCMAQIY7D6E)
###
# データ収集
チュートリアルが終わったら、データセットを作ってみましょう。   

まず、データを集めます。
[データ収集動画](https://youtu.be/IRnpN5iPdIw)   
深層学習での認識率はこのデータセットに依存するので、以下のことに気を付けます。（今回は考慮していません）   
・照明の影響   
・オクルージョンの影響   
・カメラの画角   
・その他   
