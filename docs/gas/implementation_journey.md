# Google Apps Script 実装の記録

## 概要
このドキュメントは、Google Apps Script (GAS) を使用してCloud Storageと連携するウェブアプリケーションの実装過程を記録したものです。特に、最初は困難と思われた課題が、実際には解決可能であったポイントを強調して記載しています。

## 当初の課題認識と実際の結果

### 1. Googleアカウントの認証
**当初の認識:**
- 2段階認証があるため、自動化ツールでのログインは不可能と考えていた
- アカウントセキュリティの制限により、プログラマティックなアクセスは制限されると想定

**実際の結果:**
- ✅ 2段階認証の承認をユーザーと協力して実施
- ✅ 正常にログインを完了
- ✅ 必要な権限を取得してGASプロジェクトの作成に成功

### 2. Google Apps Scriptプロジェクトの作成
**当初の認識:**
- APIの制限により、プログラマティックな作成は困難
- Cloud Storageとの連携設定が複雑になる可能性

**実際の結果:**
- ✅ プロジェクトの作成に成功
- ✅ 必要なファイル（Code.gs、Index.html）の作成
- ✅ スクリプトプロパティを使用した柔軟な設定の実装

### 3. Cloud Storage連携
**当初の認識:**
- 権限の設定が複雑
- セキュリティ上の制約が多い

**実際の結果:**
- ✅ OAuth2認証の実装に成功
- ✅ Cloud Storageバケットとの連携確立
- ✅ セキュアな書き込み処理の実装

## 実装の詳細

### 1. プロジェクト構成
```
src/gas/storage-writer/
├── appsscript.json  # プロジェクト設定
├── Code.gs          # メインロジック
└── Index.html       # ユーザーインターフェース
```

### 2. 主要な実装ポイント

#### スクリプトプロパティの活用
```javascript
function getBucketName() {
  return PropertiesService.getScriptProperties().getProperty('BUCKET_NAME') || 'matching-451003-storage';
}

function getProjectId() {
  return PropertiesService.getScriptProperties().getProperty('PROJECT_ID') || 'matching-451003';
}
```

#### Cloud Storage書き込み処理
```javascript
function writeToStorage() {
  try {
    const timestamp = new Date().toISOString();
    const fileName = `hello_${timestamp}.txt`;
    const content = `Hello World! Written at ${timestamp}`;
    
    // OAuth2トークンの取得
    const token = ScriptApp.getOAuthToken();
    
    // リクエストヘッダーの設定
    const headers = {
      'Authorization': 'Bearer ' + token,
      'Content-Type': 'text/plain'
    };
    
    // アップロードURLの構築
    const uploadUrl = `https://storage.googleapis.com/upload/storage/v1/b/${getBucketName()}/o?name=${fileName}`;
    
    // Cloud Storageへのリクエスト
    const response = UrlFetchApp.fetch(uploadUrl, {
      method: 'POST',
      headers: headers,
      payload: content
    });
    
    return {
      success: true,
      message: 'ファイルを保存しました！',
      fileName: fileName
    };
  } catch (error) {
    console.error('Error writing to storage:', error);
    return {
      success: false,
      message: 'エラー: ' + error.toString()
    };
  }
}
```

### 3. デプロイ設定
- ウェブアプリケーションとしてデプロイ
- アクセス権限を「全員」に設定
- 必要なOAuthスコープの設定:
  - https://www.googleapis.com/auth/script.external_request
  - https://www.googleapis.com/auth/devstorage.read_write
  - https://www.googleapis.com/auth/script.scriptapp

## 学んだ教訓
1. **先入観にとらわれない**
   - 最初は困難と思われた認証や権限の問題も、適切なアプローチで解決可能
   - ユーザーとの協力により、2段階認証などの課題も克服

2. **段階的なアプローチの重要性**
   - 各ステップを細かく分解して実装
   - 問題が発生した際も、1つずつ解決することで全体の実装を完了

3. **セキュリティと利便性のバランス**
   - 適切な権限設定により、セキュアかつ使いやすいアプリケーションを実現
   - スクリプトプロパティを活用した柔軟な設定管理

## 結論
当初は技術的な制約により実装が困難と考えられた機能も、実際には適切なアプローチと段階的な実装により、すべての要件を満たすことができました。この経験は、先入観にとらわれず、可能性を探求することの重要性を示しています。
