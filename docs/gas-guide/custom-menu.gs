// カスタムメニューの作成と実用的な機能の実装
function onOpen() {
  var ui = SpreadsheetApp.getUi();
  ui.createMenu('🔧 ユーティリティ')
    .addSubMenu(ui.createMenu('📝 データ操作')
      .addItem('選択したセルの合計を表示', 'showSum')
      .addItem('選択範囲をクリア', 'clearSelection')
      .addSeparator()
      .addItem('データを降順で並べ替え', 'sortDescending')
      .addItem('データを昇順で並べ替え', 'sortAscending'))
    .addSeparator()
    .addSubMenu(ui.createMenu('🎨 フォーマット')
      .addItem('選択範囲に罫線を追加', 'addBorders')
      .addItem('選択範囲の書式をクリア', 'clearFormat')
      .addSeparator()
      .addItem('ヘッダー行として設定', 'formatAsHeader'))
    .addSeparator()
    .addSubMenu(ui.createMenu('📊 分析')
      .addItem('基本統計を表示', 'showStatistics')
      .addItem('グラフを作成', 'createChart'))
    .addToUi();
}

// データ操作関数
function showSum() {
  var range = SpreadsheetApp.getActiveRange();
  var sum = 0;
  var values = range.getValues();
  
  for (var i = 0; i < values.length; i++) {
    for (var j = 0; j < values[i].length; j++) {
      if (typeof values[i][j] === 'number') {
        sum += values[i][j];
      }
    }
  }
  
  SpreadsheetApp.getUi().alert('選択範囲の合計: ' + sum);
}

function clearSelection() {
  SpreadsheetApp.getActiveRange().clear();
}

function sortDescending() {
  var range = SpreadsheetApp.getActiveRange();
  range.sort({column: 1, ascending: false});
}

function sortAscending() {
  var range = SpreadsheetApp.getActiveRange();
  range.sort({column: 1, ascending: true});
}

// フォーマット関数
function addBorders() {
  var range = SpreadsheetApp.getActiveRange();
  range.setBorder(true, true, true, true, true, true);
}

function clearFormat() {
  SpreadsheetApp.getActiveRange().clearFormat();
}

function formatAsHeader() {
  var range = SpreadsheetApp.getActiveRange();
  range.setBackground('#f3f3f3')
    .setFontWeight('bold')
    .setBorder(true, true, true, true, true, true, 'black', SpreadsheetApp.BorderStyle.SOLID);
}

// 分析関数
function showStatistics() {
  var range = SpreadsheetApp.getActiveRange();
  var values = range.getValues().flat().filter(function(value) {
    return typeof value === 'number';
  });
  
  if (values.length === 0) {
    SpreadsheetApp.getUi().alert('数値データが見つかりません。');
    return;
  }
  
  var sum = values.reduce(function(a, b) { return a + b; }, 0);
  var avg = sum / values.length;
  var max = Math.max.apply(null, values);
  var min = Math.min.apply(null, values);
  
  var message = '基本統計:\n' +
                '合計: ' + sum + '\n' +
                '平均: ' + avg + '\n' +
                '最大値: ' + max + '\n' +
                '最小値: ' + min + '\n' +
                'データ数: ' + values.length;
  
  SpreadsheetApp.getUi().alert(message);
}

function createChart() {
  var sheet = SpreadsheetApp.getActiveSheet();
  var range = SpreadsheetApp.getActiveRange();
  
  var chart = sheet.newChart()
    .setChartType(Charts.ChartType.COLUMN)
    .addRange(range)
    .setPosition(5, 5, 0, 0)
    .build();
  
  sheet.insertChart(chart);
}
