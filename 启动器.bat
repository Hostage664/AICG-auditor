@echo off
chcp 65001 >nul
title 图书馆AIGC内容审核工具

:: 获取拖拽的文件或文件夹
set "INPUT_PATH=%~1"

if "%INPUT_PATH%"=="" (
    echo 用法：将 TXT 文件或文件夹拖拽到本 BAT 文件上
    echo.
    pause
    exit /b 1
)

:: 查找 Python（优先 python3.11，其次 python，最后 py）
set "PYTHON_CMD="
for %%C in (python3.11 python3 python py) do (
    %%C --version >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_CMD=%%C"
        goto :found_python
    )
)

:: 还是没找到
echo [错误] 未找到 Python，请先安装 Python 3.8+
echo.
echo 尝试以下解决方案：
echo 1. 安装 Python 时勾选 "Add Python to PATH"
echo 2. 或手动添加 Python 路径到系统环境变量
echo 3. 或修改本 BAT 文件中的 PYTHON_CMD 为绝对路径
echo.
pause
exit /b 1

:found_python
echo [信息] 使用 Python: %PYTHON_CMD%
%PYTHON_CMD% --version

:: 检查依赖
%PYTHON_CMD% -c "import numpy, matplotlib, faiss, sentence_transformers, numba" >nul 2>&1
if errorlevel 1 (
    echo [提示] 正在安装依赖...
    %PYTHON_CMD% -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 (
        echo [错误] 依赖安装失败
        pause
        exit /b 1
    )
)

:: 切换到脚本所在目录
cd /d "%~dp0"

:: 创建输出目录
if not exist output mkdir output

:: 运行审核
echo.
echo ===========================================
echo 图书馆AIGC内容审核工具
echo ===========================================
echo 输入：%INPUT_PATH%
echo.

%PYTHON_CMD% src/batch_audit.py "%INPUT_PATH%"

echo.
echo ===========================================
echo 审核完成，结果保存在 output/ 目录
echo ===========================================
pause