// Test cases for Hello World parameter handling
const testCases = [
  {
    name: 'Default case - no parameter',
    input: {},
    expected: 'Hello World!'
  },
  {
    name: 'Japanese name parameter',
    input: { parameter: { name: '太郎' } },
    expected: 'Hello 太郎!'
  },
  {
    name: 'English name parameter',
    input: { parameter: { name: 'John' } },
    expected: 'Hello John!'
  }
];

// Mock HtmlService
const HtmlService = {
  createHtmlOutput: (text) => text
};

// Import the doGet function
function doGet(e) {
  var name = (e && e.parameter && e.parameter.name) || 'World';
  return HtmlService.createHtmlOutput('Hello ' + name + '!');
}

// Run tests
console.log('Running Hello World parameter tests...\n');
testCases.forEach((test, index) => {
  const actual = doGet(test.input);
  const passed = actual === test.expected;
  console.log(`Test ${index + 1}: ${test.name}`);
  console.log(`Expected: ${test.expected}`);
  console.log(`Actual: ${actual}`);
  console.log(`Result: ${passed ? 'PASSED' : 'FAILED'}\n`);
});
