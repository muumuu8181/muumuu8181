/**
 * パラメーターを受け取ってカスタマイズされた挨拶を返す
 * @param {Object} e - リクエストパラメーター
 * @return {HtmlOutput} カスタマイズされた挨拶のHTML
 */
function doGet(e) {
  var name = e.parameter.name || 'World';
  return HtmlService.createHtmlOutput('Hello ' + name + '!');
}
