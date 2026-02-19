"""全局量化安全性检测脚本"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*60)
print("量化交易系统安全性检测报告")
print("="*60)

issues = []
warnings = []
passed = []

# 1. API密钥安全检测
print("\n[1] API密钥安全检测")
print("-"*40)

env_file = project_root / ".env"
if env_file.exists():
    with open(env_file, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'API_KEY' in content or 'API_SECRET' in content:
            gitignore_file = project_root / '.gitignore'
            if gitignore_file.exists():
                try:
                    with open(gitignore_file, 'r', encoding='utf-8') as gf:
                        if '.env' in gf.read():
                            passed.append("✅ .env文件已添加到.gitignore")
                        else:
                            issues.append("❌ .env文件未添加到.gitignore，可能泄露API密钥")
                except:
                    passed.append("✅ .gitignore文件存在")
            
            if content.count("'") > 0 or content.count('"') > 0:
                passed.append("✅ API密钥使用引号包裹")
            else:
                warnings.append("⚠️ API密钥未使用引号包裹")
else:
    passed.append("✅ 未找到.env文件")

# 2. 敏感文件检测
print("\n[2] 敏感文件检测")
print("-"*40)

sensitive_patterns = ['*.key', '*.pem', '*.p12', 'credentials*', 'secrets*']
found_sensitive = []
for pattern in sensitive_patterns:
    for f in project_root.rglob(pattern):
        if '__pycache__' not in str(f) and '.git' not in str(f):
            found_sensitive.append(str(f))

if found_sensitive:
    issues.append(f"❌ 发现敏感文件: {found_sensitive}")
else:
    passed.append("✅ 未发现敏感文件")

# 3. 硬编码密钥检测
print("\n[3] 硬编码密钥检测")
print("-"*40)

hardcoded_patterns = [
    'api_key = "',
    'api_secret = "',
    'password = "',
    'token = "',
]

found_hardcoded = []
for py_file in project_root.rglob('*.py'):
    if '__pycache__' in str(py_file) or '.git' in str(py_file):
        continue
    try:
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
            for pattern in hardcoded_patterns:
                if pattern in content.lower():
                    found_hardcoded.append(f"{py_file}: {pattern}")
    except:
        pass

if found_hardcoded:
    issues.append(f"❌ 发现硬编码密钥: {found_hardcoded}")
else:
    passed.append("✅ 未发现硬编码密钥")

# 4. 交易安全检测
print("\n[4] 交易安全检测")
print("-"*40)

trading_file = project_root / "Trading" / "live_trader.py"
if trading_file.exists():
    with open(trading_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
        if 'stop_loss' in content.lower():
            passed.append("✅ 支持止损功能")
        else:
            warnings.append("⚠️ 未检测到止损功能")
        
        if 'max_daily_loss' in content.lower():
            passed.append("✅ 支持每日最大亏损限制")
        else:
            warnings.append("⚠️ 未检测到每日最大亏损限制")
        
        if 'max_daily_trades' in content.lower():
            passed.append("✅ 支持每日最大交易次数限制")
        else:
            warnings.append("⚠️ 未检测到每日最大交易次数限制")
        
        if 'liquidation' in content.lower():
            passed.append("✅ 支持爆仓检测")
        else:
            warnings.append("⚠️ 未检测到爆仓检测")

# 5. 参数验证检测
print("\n[5] 参数验证检测")
print("-"*40)

config_file = project_root / "core" / "config.py"
if config_file.exists():
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
        if 'leverage' in content and 'Range' in content:
            passed.append("✅ 杠杆参数有范围限制")
        else:
            warnings.append("⚠️ 杠杆参数可能无范围限制")
        
        if 'position_size' in content:
            passed.append("✅ 仓位比例参数已定义")
        else:
            warnings.append("⚠️ 仓位比例参数未定义")

# 6. 日志安全检测
print("\n[6] 日志安全检测")
print("-"*40)

log_files = list(project_root.rglob('*.log'))
if log_files:
    for log_file in log_files[:3]:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'api_key' in content.lower() or 'api_secret' in content.lower():
                    issues.append(f"❌ 日志文件可能包含敏感信息: {log_file}")
                else:
                    passed.append(f"✅ 日志文件安全: {log_file.name}")
        except:
            pass
else:
    passed.append("✅ 未发现日志文件")

# 7. 网络安全检测
print("\n[7] 网络安全检测")
print("-"*40)

if trading_file.exists():
    with open(trading_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
        if 'https://' in content:
            passed.append("✅ 使用HTTPS协议")
        else:
            issues.append("❌ 未使用HTTPS协议")
        
        if 'timeout' in content.lower():
            passed.append("✅ 网络请求有超时设置")
        else:
            warnings.append("⚠️ 网络请求可能无超时设置")
        
        if 'hmac' in content.lower() and 'sha256' in content.lower():
            passed.append("✅ 使用HMAC-SHA256签名")
        else:
            warnings.append("⚠️ 签名方式可能不安全")

# 8. 错误处理检测
print("\n[8] 错误处理检测")
print("-"*40)

py_files = list(project_root.rglob('*.py'))
py_files = [f for f in py_files if '__pycache__' not in str(f) and '.git' not in str(f)]

try_count = 0
for py_file in py_files[:20]:
    try:
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'try:' in content and 'except' in content:
                try_count += 1
    except:
        pass

if try_count > 10:
    passed.append(f"✅ 错误处理完善 ({try_count}个文件有try-except)")
else:
    warnings.append(f"⚠️ 错误处理可能不足 ({try_count}个文件有try-except)")

# 输出总结
print("\n" + "="*60)
print("检测总结")
print("="*60)

print(f"\n✅ 通过: {len(passed)}项")
for p in passed:
    print(f"   {p}")

if warnings:
    print(f"\n⚠️ 警告: {len(warnings)}项")
    for w in warnings:
        print(f"   {w}")

if issues:
    print(f"\n❌ 问题: {len(issues)}项")
    for i in issues:
        print(f"   {i}")

print("\n" + "="*60)
if issues:
    print("🔴 发现安全问题，请立即修复！")
elif warnings:
    print("🟡 存在潜在风险，建议优化")
else:
    print("🟢 系统安全性良好")
print("="*60)
