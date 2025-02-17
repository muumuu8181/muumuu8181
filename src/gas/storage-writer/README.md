# GAS Storage Writer

## セットアップ手順 (Setup Instructions)

1. [Google Apps Script](https://script.google.com/) にアクセス
2. 新しいプロジェクトを作成
3. `Code.gs` と `Index.html` の内容をコピー
4. デプロイ → 新しいデプロイ → ウェブアプリケーション
   - 次のユーザーとして実行: 自分
   - アクセスできるユーザー: 全員
5. 「許可を確認」をクリック
6. アプリをデプロイ
7. デプロイ後のURLをiPhoneのブックマークに追加

## 必要な権限 (Required Permissions)
- Google Cloud Storage API
- Google Apps Script OAuth

## 使用方法 (Usage)
1. ブックマークしたURLを開く
2. 「保存する」ボタンをタップ
3. Cloud Storageにテキストファイルが保存される
