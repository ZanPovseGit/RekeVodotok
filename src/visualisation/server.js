const express = require('express');
const http = require('http');
const fs = require('fs');
const path = require('path');
const cors = require('cors');

const app = express();

app.use(cors());

app.use(express.static('public', {
  setHeaders: (res, filePath) => {
    const contentType = getFileContentType(filePath);
    if (contentType) {
      res.setHeader('Content-Type', contentType);
    }
  }
}));

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

const server = http.createServer(app);
server.listen(3001, () => {
  console.log('Server is running on http://localhost:3001');
});

function getFileContentType(filePath) {
  const extname = path.extname(filePath);
  switch (extname) {
    case '.css':
      return 'text/css';
    case '.js':
      return 'application/javascript';
    case '.html':
      return 'text/html';
    default:
      return 'text/plain'; 
  }
}
