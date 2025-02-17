// Cloud Storage bucket configuration
const BUCKET_NAME = 'matching-451003.appspot.com';
const PROJECT_ID = 'matching-451003';

// Create the web interface
function doGet() {
  const template = HtmlService.createTemplateFromFile('Index');
  return template.evaluate()
    .setTitle('Storage Writer')
    .setFaviconUrl('https://www.google.com/images/favicon.ico')
    .addMetaTag('viewport', 'width=device-width, initial-scale=1');
}

// Handle the button click
function writeToStorage() {
  try {
    const timestamp = new Date().toISOString();
    const fileName = `hello_${timestamp}.txt`;
    const content = `Hello World! Written at ${timestamp}`;
    
    // Get OAuth2 token
    const token = ScriptApp.getOAuthToken();
    
    // Create headers for the request
    const headers = {
      'Authorization': 'Bearer ' + token,
      'Content-Type': 'text/plain'
    };
    
    // Construct the upload URL
    const uploadUrl = `https://storage.googleapis.com/upload/storage/v1/b/${BUCKET_NAME}/o?name=${fileName}`;
    
    // Make the request to Cloud Storage
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
