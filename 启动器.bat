@echo off
chcp 65001 >nul
title 图书馆 AIGC 内容审核工具

:: ═══════════════════════════════════════════════════════════
::  启动器.bat · 重构版
::  用法：将 .txt 文件或含 .txt 的文件夹拖拽到本文件上
::  依赖：Python 3.8+，requirements.txt 中的包
:: ═══════════════════════════════════════════════════════════

echo.
echo ===========================================
echo  图书馆 AIGC 内容审核工具
echo ===========================================
echo.

:: ── Step 0：获取拖拽输入 ──────────────────────────────────
set "INPUT_PATH=%~1"

if "%INPUT_PATH%"=="" (
    echo [错误] 未检测到输入路径
    echo.
    echo  用法：将 TXT 文件 或 含 TXT 的文件夹
    echo        拖拽到本 BAT 文件图标上运行
    echo.
    pause
    exit /b 1
)

echo [输入] %INPUT_PATH%
echo.

:: ── Step 1：查找 Python（优先高版本）────────────────────────
set "PYTHON_CMD="

for %%C in (python3.11 python3.10 python3.9 python3.8 python3 python py) do (
    %%C --version >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_CMD=%%C"
        goto :found_python
    )
)

:: 找不到 Python
echo [错误] 未检测到可用的 Python 解释器
echo.
echo  解决方案：
echo    1. 访问 https://www.python.org/downloads/ 安装 Python 3.8+
echo    2. 安装时勾选 "Add Python to PATH"
echo    3. 或将 Python 安装目录手动加入系统环境变量 PATH
echo    4. 或直接修改本 BAT 文件第一行为 Python 绝对路径：
echo       set "PYTHON_CMD=C:\Python311\python.exe"
echo.
pause
exit /b 1

:found_python
echo [Python] 检测到: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

:: ── Step 2：校验 Python 版本 >= 3.8 ─────────────────────────
%PYTHON_CMD% -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [错误] Python 版本过低，需要 3.8 或更高版本
    echo        当前版本:
    %PYTHON_CMD% --version
    echo.
    pause
    exit /b 1
)

:: ── Step 3：检测虚拟环境（可选提示）─────────────────────────
if exist "%~dp0.venv\Scripts\python.exe" (
    echo [提示] 检测到项目虚拟环境 .venv，建议使用虚拟环境运行
    echo        如需启用：在命令行执行 .venv\Scripts\activate 后再运行
    echo.
)

:: ── Step 4：检查核心依赖 ──────────────────────────────────────
echo [依赖] 正在检查依赖...
%PYTHON_CMD% -c ^
    "import numpy, matplotlib, scipy, faiss, sentence_transformers, requests" ^
    >nul 2>&1

if errorlevel 1 (
    echo [依赖] 检测到缺失依赖，正在安装 requirements.txt...
    echo.
    %PYTHON_CMD% -m pip install -r "%~dp0requirements.txt" --timeout 120
    if errorlevel 1 (
        echo.
        echo [错误] 依赖安装失败，可能原因：
        echo    1. 网络连接问题 → 尝试切换到国内镜像：
        echo       pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
        echo    2. pip 版本过低 → 先执行：python -m pip install --upgrade pip
        echo    3. torch 安装超时 → 手动安装后重试
        echo.
        pause
        exit /b 1
    )
    echo.
    echo [依赖] 安装完成
    echo.
) else (
    echo [依赖] 全部就绪
    echo.
)

:: ── Step 5：初始化目录结构 ────────────────────────────────────
cd /d "%~dp0"

if not exist output         mkdir output
if not exist output\charts  mkdir output\charts
if not exist config         mkdir config

:: ── Step 6：运行审核主程序 ────────────────────────────────────
echo ===========================================
echo  开始审核
echo  输入路径: %INPUT_PATH%
echo ===========================================
echo.

%PYTHON_CMD% src\batch_audit.py "%INPUT_PATH%"
set "EXIT_CODE=%errorlevel%"

echo.

:: ── Step 7：退出状态提示 ──────────────────────────────────────
if %EXIT_CODE% == 0 (
    echo ===========================================
    echo  审核完成
    echo  结果保存在 output\ 目录
    echo  汇总报告: output\audit_summary.json
    echo  运行日志: output\audit.log
    echo ===========================================
) else (
    echo ===========================================
    echo  [警告] 审核程序异常退出，退出码: %EXIT_CODE%
    echo  请检查 output\audit.log 获取详细错误信息
    echo ===========================================
)

echo.
pause
exit /b %EXIT_CODE%