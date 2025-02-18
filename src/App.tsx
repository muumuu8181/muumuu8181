import { useState, useRef, useEffect } from "react";
import "./App.css";
import { SEND_TIMEOUT, UPLOAD_TIMEOUT, MAX_RETRIES, ERROR_MESSAGES } from "./constants";

const API_URL = import.meta.env.VITE_API_URL;
interface Message {
  id: string;
  text: string;
  timestamp: string;
}

interface File {
  id: string;
  name: string;
  size: string;
  timestamp: string;
}

interface DashboardProps {
  onLogout: () => void;
  token: string | null;
}



const Dashboard: React.FC<DashboardProps> = ({ onLogout, token }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [files, setFiles] = useState<File[]>([]);
  const [newMessage, setNewMessage] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const fetchMessages = async () => {
    try {
      const response = await fetch(`${API_URL}/api/messages`, {
        mode: 'cors',
        headers: {
          'Accept': 'application/json',
        }
      });
      if (!response.ok) throw new Error("メッセージの取得に失敗しました");
      const data = await response.json();
      setMessages(data);
    } catch (err) {
      console.error("Error fetching messages:", err);
      setError("メッセージの取得に失敗しました");
    }
  };

  const fetchFiles = async () => {
    try {
      const response = await fetch(`${API_URL}/api/files`, {
        mode: 'cors',
        headers: {
          'Accept': 'application/json',
        }
      });
      if (!response.ok) throw new Error("ファイル一覧の取得に失敗しました");
      const data = await response.json();
      setFiles(data);
    } catch (err) {
      console.error("Error fetching files:", err);
      setError("ファイル一覧の取得に失敗しました");
    }
  };

  const handleSendMessage = async (retryCount = 0) => {
    if (!newMessage.trim()) return;
    
    if (retryCount >= MAX_RETRIES) {
      setError(ERROR_MESSAGES.maxRetries);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    const timeoutId = setTimeout(() => {
      setIsLoading(false);
      setError(ERROR_MESSAGES.timeout);
      handleSendMessage(retryCount + 1);
    }, SEND_TIMEOUT);

      try {
        const formData = new FormData();
        formData.append("text", newMessage);

        const controller = new AbortController();
        const response = await fetch(`${API_URL}/api/messages`, {
          method: "POST",
          body: formData,
          signal: controller.signal,
          mode: 'cors',
          headers: {
            'Accept': 'application/json'
          }
        });

        if (!response.ok) throw new Error("メッセージの送信に失敗しました");
        
        const message = await response.json();
        setMessages([message, ...messages]);
        setNewMessage("");
        setError("");
        setSuccess("メッセージを送信しました");
        setTimeout(() => setSuccess(""), 3000);
      } catch (err) {
        console.error("Error sending message:", err);
        if (err instanceof TypeError && err.message.includes('fetch')) {
          setError(ERROR_MESSAGES.network);
        } else if (retryCount < MAX_RETRIES - 1) {
          setError(`メッセージの送信に失敗しました。再試行中... (${retryCount + 1}/${MAX_RETRIES})`);
        } else {
          setError("メッセージの送信に失敗しました。もう一度お試しください。");
        }
      } finally {
        clearTimeout(timeoutId);
        setIsLoading(false);
      }
  };

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = event.target.files?.[0];
    if (uploadedFile) {
      if (uploadedFile.size > 5 * 1024 * 1024) {
        setError("ファイルサイズは5MB以下にしてください");
        setSelectedFile(null);
        return;
      }
      setSelectedFile({
        id: "temp",
        name: uploadedFile.name,
        size: `${(uploadedFile.size / 1024 / 1024).toFixed(2)}MB`,
        timestamp: new Date().toLocaleString()
      });
      setError("");
    }
  };

  const handleFileUpload = async (retryCount = 0) => {
    if (!selectedFile) return;
    
    if (retryCount >= MAX_RETRIES) {
      setError(ERROR_MESSAGES.maxRetries);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    const timeoutId = setTimeout(() => {
      setIsLoading(false);
      setError(ERROR_MESSAGES.timeout);
      handleFileUpload(retryCount + 1);
    }, UPLOAD_TIMEOUT);

    try {
      const uploadedFile = fileInputRef.current?.files?.[0];
      if (!uploadedFile) throw new Error("ファイルが選択されていません");

      const formData = new FormData();
      formData.append("file", uploadedFile);

      const controller = new AbortController();
      const response = await fetch(`${API_URL}/api/files`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
        mode: 'cors',
        headers: {
          'Accept': 'application/json'
        }
      });

      if (!response.ok) throw new Error("ファイルのアップロードに失敗しました");
      
      const fileInfo = await response.json();
      setFiles([fileInfo, ...files]);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      setError("");
      setSuccess("ファイルをアップロードしました");
      setTimeout(() => setSuccess(""), 3000);
    } catch (err) {
      console.error("Error uploading file:", err);
      if (err instanceof TypeError && err.message.includes('fetch')) {
        setError(ERROR_MESSAGES.network);
      } else if (retryCount < MAX_RETRIES - 1) {
        setError(`ファイルのアップロードに失敗しました。再試行中... (${retryCount + 1}/${MAX_RETRIES})`);
      } else {
        setError("ファイルのアップロードに失敗しました。もう一度お試しください。");
      }
    } finally {
      clearTimeout(timeoutId);
      setIsLoading(false);
    }
  };

  const downloadFile = async (fileId: string, fileName: string) => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/files/${fileId}`, {
        mode: 'cors',
        headers: {
          'Accept': 'application/json'
        }
      });
      if (!response.ok) throw new Error("ファイルのダウンロードに失敗しました");
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = fileName;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      setSuccess("ファイルをダウンロードしました");
      setTimeout(() => setSuccess(""), 3000);
    } catch (err) {
      console.error("Error downloading file:", err);
      setError("ファイルのダウンロードに失敗しました");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (!token) {
      setError("認証が必要です。ログインしてください。");
      return;
    }
    
    fetchMessages();
    fetchFiles();
  }, [token]);

  return (
    <div className="dashboard">
      <h2>ダッシュボード画面</h2>
      {error && <div className="error-message">{error}</div>}
      {success && <div className="success-message">{success}</div>}
      {isLoading && <div className="loading-message">処理中...</div>}
      <div className="dashboard-content">
        <div className="dashboard-section">
          <h3>メッセージ一覧</h3>
          <div className="message-input">
            <input
              type="text"
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              placeholder="新しいメッセージを入力..."
              className="message-text-input"
              disabled={isLoading}
            />
            <button 
              onClick={() => handleSendMessage(0)} 
              className="send-button"
              disabled={isLoading || !newMessage.trim()}
            >
              送信
            </button>
          </div>
          <div className="message-list">
            {messages.map((message) => (
              <div key={message.id} className="message">
                <p>{message.text}</p>
                <small>{message.timestamp}</small>
              </div>
            ))}
          </div>
        </div>
        <div className="dashboard-section">
          <h3>ファイル一覧</h3>
          <div className="file-upload">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              style={{ display: "none" }}
              disabled={isLoading}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="upload-button"
              disabled={isLoading}
            >
              ファイルを選択
            </button>
            {selectedFile && (
              <div className="selected-file">
                <p>{selectedFile.name} ({selectedFile.size})</p>
                <button 
                  onClick={() => handleFileUpload(0)}
                  className="upload-button"
                  disabled={isLoading}
                >
                  アップロード
                </button>
              </div>
            )}
            <small className="upload-hint">※ 5MB以下のファイル</small>
          </div>
          <div className="file-list">
            {files.map((file) => (
              <div key={file.id} className="file">
                <p onClick={() => downloadFile(file.id, file.name)} style={{ cursor: "pointer" }}>
                  {file.name}
                </p>
                <small>
                  サイズ: {file.size} • {file.timestamp}
                </small>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="action-buttons">
        <button onClick={onLogout} className="logout-button" disabled={isLoading}>
          ログアウト
        </button>
      </div>
    </div>
  );
};

function App() {
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleTestLogin = async () => {
    setIsLoading(true);
    try {
      localStorage.setItem("auth_token", "mock_token_for_testing");
      setToken("mock_token_for_testing");
      setError("");
      setMessage("テストユーザーとしてログインしました！");
    } catch (err) {
      console.error("Test login error:", err);
      setError(ERROR_MESSAGES.auth);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("auth_token");
    setToken(null);
    setMessage("ログアウトしました。");
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Azure認証デモ</h1>
        {error && <div className="error-message">{error}</div>}
        {message && <div className="success-message">{message}</div>}
        {isLoading && <div className="loading-message">処理中...</div>}
        
        {!token ? (
          <div className="auth-options">
            <div className="auth-section">
              <h2>テストユーザーとしてログイン</h2>
              <button 
                onClick={handleTestLogin} 
                className="test-login"
                disabled={isLoading}
              >
                テストログインボタン
              </button>
              <p className="hint">※ このボタンをクリックすると、テストユーザーとしてログインできます</p>
            </div>
          </div>
        ) : (
          <Dashboard onLogout={handleLogout} token={token} />
        )}
      </header>
    </div>
  );
}

export default App;
