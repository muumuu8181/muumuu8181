export const SEND_TIMEOUT = 60000;  // 60 seconds for cold starts
export const UPLOAD_TIMEOUT = 60000;
export const MAX_RETRIES = 5;
export const RETRY_DELAY = 2000;  // 2 seconds between retries
export const ERROR_MESSAGES = {
  timeout: "サーバーの起動中です。しばらくお待ちください...",
  maxRetries: "サーバーの応答がありません。時間をおいて再度お試しください。",
  auth: "認証に失敗しました。もう一度ログインしてください。",
  network: "ネットワークエラーが発生しました。接続を確認してください。"
};
