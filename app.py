from flask import Flask, render_template_string, send_file
import os

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>3D Game</title>
    <style>
        body { margin: 0; padding: 20px; text-align: center; font-family: Arial, sans-serif; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #333; }
        .instructions { 
            background: #f5f5f5; 
            padding: 20px; 
            border-radius: 8px;
            margin: 20px 0;
            text-align: left;
        }
        .download-btn {
            display: inline-block;
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin-top: 20px;
        }
        .download-btn:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>3D Game</h1>
        <div class="instructions">
            <h2>ゲーム説明:</h2>
            <ul>
                <li>移動: 矢印キーまたはWASD</li>
                <li>ジャンプ: スペースキー</li>
                <li>攻撃: Xキー</li>
            </ul>
            <h2>ゲームの目的:</h2>
            <ul>
                <li>赤い敵を倒してスコアを稼ぐ</li>
                <li>敵との接触を避けて生存する</li>
                <li>残機がなくなるとゲームオーバー</li>
            </ul>
        </div>
        <a href="/download" class="download-btn">ゲームをダウンロード</a>
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/download')
def download():
    return send_file('game.exe', as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
