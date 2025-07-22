#!/usr/bin/env node

/**
 * 前端启动脚本
 * 
 * 这个脚本用于启动 Vue.js 前端开发服务器。
 * 它会自动检查 Node.js 版本、安装依赖，并启动开发服务器。
 */

const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

// 颜色输出函数
const colors = {
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  white: '\x1b[37m',
  reset: '\x1b[0m'
};

function colorLog(color, message) {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function checkNodeVersion() {
  const nodeVersion = process.version;
  const majorVersion = parseInt(nodeVersion.slice(1).split('.')[0]);
  
  if (majorVersion < 18) {
    colorLog('red', '❌ 错误: 需要 Node.js 18 或更高版本');
    colorLog('red', `当前版本: ${nodeVersion}`);
    process.exit(1);
  }
  
  colorLog('green', `✅ Node.js 版本: ${nodeVersion}`);
}

function checkPackageManager() {
  // 检查是否有 package-lock.json 或 yarn.lock
  const frontendDir = path.join(__dirname, 'frontend');
  const hasPackageLock = fs.existsSync(path.join(frontendDir, 'package-lock.json'));
  const hasYarnLock = fs.existsSync(path.join(frontendDir, 'yarn.lock'));
  
  if (hasYarnLock) {
    return 'yarn';
  } else if (hasPackageLock) {
    return 'npm';
  } else {
    // 检查系统中是否有 yarn
    try {
      execSync('yarn --version', { stdio: 'ignore' });
      return 'yarn';
    } catch {
      return 'npm';
    }
  }
}

function checkDependencies() {
  const frontendDir = path.join(__dirname, 'frontend');
  const packageJsonPath = path.join(frontendDir, 'package.json');
  
  if (!fs.existsSync(packageJsonPath)) {
    colorLog('red', '❌ 错误: package.json 文件不存在');
    process.exit(1);
  }
  
  const nodeModulesPath = path.join(frontendDir, 'node_modules');
  
  if (!fs.existsSync(nodeModulesPath)) {
    colorLog('yellow', '📦 node_modules 不存在，需要安装依赖...');
    return false;
  }
  
  // 检查关键依赖是否存在
  const keyDeps = ['vue', '@ai-sdk/vue', 'vite'];
  for (const dep of keyDeps) {
    const depPath = path.join(nodeModulesPath, dep);
    if (!fs.existsSync(depPath)) {
      colorLog('yellow', `📦 缺少依赖 ${dep}，需要重新安装...`);
      return false;
    }
  }
  
  colorLog('green', '✅ 所有依赖已安装');
  return true;
}

function installDependencies(packageManager) {
  const frontendDir = path.join(__dirname, 'frontend');
  
  colorLog('blue', `📦 使用 ${packageManager} 安装依赖...`);
  
  try {
    const installCmd = packageManager === 'yarn' ? 'yarn install' : 'npm install';
    execSync(installCmd, { 
      cwd: frontendDir, 
      stdio: 'inherit',
      env: { ...process.env, NODE_ENV: 'development' }
    });
    colorLog('green', '✅ 依赖安装完成');
  } catch (error) {
    colorLog('red', `❌ 依赖安装失败: ${error.message}`);
    process.exit(1);
  }
}

function checkBackendStatus() {
  return new Promise((resolve) => {
    const http = require('http');
    
    const req = http.get('http://localhost:8000/health', (res) => {
      resolve(res.statusCode === 200);
    });
    
    req.on('error', () => {
      resolve(false);
    });
    
    req.setTimeout(2000, () => {
      req.destroy();
      resolve(false);
    });
  });
}

function startDevServer(packageManager) {
  const frontendDir = path.join(__dirname, 'frontend');
  
  colorLog('cyan', '🚀 启动 Vue.js 开发服务器...');
  colorLog('cyan', `📁 工作目录: ${frontendDir}`);
  colorLog('cyan', '🌐 前端地址: http://localhost:3000');
  colorLog('cyan', '🔗 后端代理: http://localhost:8000');
  colorLog('cyan', '\n按 Ctrl+C 停止服务器\n');
  
  const devCmd = packageManager === 'yarn' ? 'yarn' : 'npm';
  const devArgs = packageManager === 'yarn' ? ['dev'] : ['run', 'dev'];
  
  const child = spawn(devCmd, devArgs, {
    cwd: frontendDir,
    stdio: 'inherit',
    env: { 
      ...process.env, 
      NODE_ENV: 'development',
      FORCE_COLOR: '1'
    },
    shell: os.platform() === 'win32'
  });
  
  child.on('error', (error) => {
    colorLog('red', `❌ 启动开发服务器失败: ${error.message}`);
    process.exit(1);
  });
  
  child.on('exit', (code) => {
    if (code !== 0 && code !== null) {
      colorLog('red', `❌ 开发服务器异常退出，代码: ${code}`);
    } else {
      colorLog('green', '👋 开发服务器已停止');
    }
  });
  
  // 处理进程终止信号
  process.on('SIGINT', () => {
    colorLog('yellow', '\n🛑 正在停止开发服务器...');
    child.kill('SIGINT');
  });
  
  process.on('SIGTERM', () => {
    child.kill('SIGTERM');
  });
}

async function main() {
  colorLog('magenta', '🔥 LangChain AI SDK Adapter - 前端启动器');
  colorLog('white', '='.repeat(50));
  
  try {
    // 检查 Node.js 版本
    checkNodeVersion();
    
    // 检查包管理器
    const packageManager = checkPackageManager();
    colorLog('blue', `📦 使用包管理器: ${packageManager}`);
    
    // 检查依赖
    const depsInstalled = checkDependencies();
    if (!depsInstalled) {
      installDependencies(packageManager);
    }
    
    // 检查后端状态
    colorLog('blue', '🔍 检查后端服务状态...');
    const backendRunning = await checkBackendStatus();
    if (backendRunning) {
      colorLog('green', '✅ 后端服务正在运行');
    } else {
      colorLog('yellow', '⚠️  后端服务未运行，请先启动后端服务');
      colorLog('yellow', '   运行: python start_backend.py');
      colorLog('yellow', '   或者: cd backend && python main.py');
    }
    
    // 启动开发服务器
    startDevServer(packageManager);
    
  } catch (error) {
    colorLog('red', `❌ 启动失败: ${error.message}`);
    process.exit(1);
  }
}

// 处理未捕获的异常
process.on('uncaughtException', (error) => {
  colorLog('red', `❌ 未捕获的异常: ${error.message}`);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  colorLog('red', `❌ 未处理的 Promise 拒绝: ${reason}`);
  process.exit(1);
});

if (require.main === module) {
  main();
}