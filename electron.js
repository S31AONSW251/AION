const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');

// Detect dev mode
const isDev = process.env.NODE_ENV === 'development';

// Keep a global reference to prevent garbage collection
let mainWindow;

// Function to check if Ollama server is running
function checkOllamaServer(callback) {
  exec('curl --silent http://localhost:11434/api/tags', (err, stdout) => {
    if (err || !stdout.includes('models')) {
      console.log('❌ Ollama not running.');
      callback(false);
    } else {
      console.log('✅ Ollama detected:', stdout);
      callback(true);
    }
  });
}

// Function to auto-start Ollama if not running
function ensureOllamaRunning() {
  checkOllamaServer((isRunning) => {
    if (!isRunning) {
      console.log('🚀 Starting Ollama server...');
      // Run Ollama in background, detached from the current process
      const ollamaProcess = exec('ollama serve', (error, stdout, stderr) => {
        if (error) {
          console.error('❌ Failed to start Ollama:', error);
        } else {
          console.log('✅ Ollama started successfully.');
        }
      });
      // Important: Detach the child process so it continues running after the parent exits
      ollamaProcess.unref();
    }
  });
}

// Create the main application window
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    webPreferences: {
      nodeIntegration: true, // Allows Node.js APIs in renderer process
      contextIsolation: false, // Disables context isolation for easier IPC (consider enabling for security)
      enableRemoteModule: true, // Enables the remote module (deprecated, consider alternative for security)
      webSecurity: false, // Disables web security (DANGER! Only for local development/testing)
      preload: path.join(__dirname, 'preload.js') // Preload script for secure IPC
    },
    icon: path.join(__dirname, 'public', 'icon.ico') // Path to your application icon
  });

  // Load the React application based on environment
  if (isDev) {
    mainWindow.loadURL('http://localhost:3000'); // Load from React dev server
    mainWindow.webContents.openDevTools(); // Open DevTools in development mode
  } else {
    mainWindow.loadFile(path.join(__dirname, 'build', 'index.html')); // Load from built React app
  }

  // Emitted when the window is closed.
  mainWindow.on('closed', () => (mainWindow = null));
}

// App ready event: ensure Ollama is running before creating the window
app.whenReady().then(() => {
  ensureOllamaRunning();
  createWindow();
});

// Quit when all windows are closed, except on macOS (Cmd+Q)
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// On macOS, re-create a window when the dock icon is clicked and no windows are open
app.on('activate', () => {
  if (mainWindow === null) createWindow();
});

// ----------------------------
// Extra APIs (like filesystem)
// ----------------------------
// IPC handler for reading files
ipcMain.handle('read-file', async (event, filePath) => {
  try {
    return fs.readFileSync(filePath, 'utf8');
  } catch (error) {
    console.error('Failed to read file:', error);
    throw error; // Re-throw to propagate error to renderer
  }
});

// IPC handler for writing files
ipcMain.handle('write-file', async (event, filePath, content) => {
  try {
    fs.writeFileSync(filePath, content, 'utf8');
    return { success: true };
  } catch (error) {
    console.error('Failed to write file:', error);
    throw error;
  }
});

// IPC handler for listing directories
ipcMain.handle('list-directory', async (event, dirPath) => {
  try {
    return fs.readdirSync(dirPath);
  } catch (error) {
    console.error('Failed to list directory:', error);
    throw error;
  }
});

// IPC handler for making HTTP requests (proxy for renderer)
ipcMain.handle('make-http-request', async (event, options) => {
  try {
    const { url, method, headers, body } = options;
    const response = await fetch(url, { method, headers, body });
    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    console.error('HTTP request failed:', error);
    throw error;
  }
});
