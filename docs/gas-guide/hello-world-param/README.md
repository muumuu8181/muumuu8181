# Google Apps Script パラメーター付きHello World ガイド

このガイドでは、パラメーターを受け取ってカスタマイズされた挨拶を返すGoogle Apps Scriptの実装方法を説明します。

## 実装コード

```javascript
function doGet(e) {
  var name = e.parameter.name || 'World';
  return HtmlService.createHtmlOutput('Hello ' + name + '!');
}
```

## 使用方法

1. Google Apps Scriptエディタでプロジェクトを作成
2. 上記のコードをコピー＆ペースト
3. デプロイボタンをクリック
4. 「新しいデプロイ」を選択
5. アクセス権を「全員」に設定
6. デプロイをクリック

## パラメーターの使用方法

デプロイ後のURLに以下のようにパラメーターを追加します：
```
https://script.google.com/macros/s/.../exec?name=あなたの名前
```

例：
- `?name=太郎` → "Hello 太郎!"
- パラメーターなし → "Hello World!"

## 注意点

- パラメーターは自動的にURLエンコードされます
- 日本語の名前も使用可能です
