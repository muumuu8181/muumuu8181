# Shared Configuration Directory

このディレクトリは全Devinプロジェクト間で共有される設定を管理します。

## 他のDevinインスタンスからのアクセス方法
```bash
# リポジトリのクローン
gh repo clone muumuu8181/muumuu8181

# 設定ディレクトリへ移動
cd muumuu8181/shared/config
```

## 構造
```
shared/
  ├── config/          # 共有設定ファイル
  │   ├── aws/        # AWS関連の設定
  │   ├── gcp/        # GCP関連の設定
  │   ├── azure/      # Azure関連の設定
  │   └── common/     # 共通設定
  └── README.md       # このファイル
```

## 使用方法
1. 上記の手順でリポジトリをクローン
2. 必要な設定を対応するディレクトリから読み込む
3. 新しい設定は適切なサブディレクトリに配置
4. 変更履歴はGitで管理

## 注意事項
- 機密情報は含めない
- プロバイダー固有の設定は対応するディレクトリに配置
- 共通設定は common ディレクトリに配置
