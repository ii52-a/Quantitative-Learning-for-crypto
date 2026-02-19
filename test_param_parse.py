"""测试参数范围解析"""
print("测试参数范围解析...")

test_text = """# 策略参数 (格式: 参数名: 最小值, 最大值, 步长)
fast_period: 5, 30, 1
slow_period: 20, 50, 2
hist_filter: 0, 5, 0.5

# 风险控制参数
leverage: 1, 20, 1
stop_loss_pct: 0, 10, 0.5
"""

from Strategy.parameter_optimizer import ParameterRange

ranges = []

for line in test_text.split('\n'):
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    
    if ':' not in line:
        continue
    
    try:
        parts = line.split(':', 1)
        name = parts[0].strip()
        value_part = parts[1].strip()
        
        if ',' in value_part:
            values = [v.strip() for v in value_part.split(',')]
            
            if len(values) == 3:
                min_val = float(values[0])
                max_val = float(values[1])
                step_val = float(values[2])
                
                if min_val == int(min_val) and max_val == int(max_val) and step_val == int(step_val):
                    ranges.append(ParameterRange(
                        name=name,
                        min_value=int(min_val),
                        max_value=int(max_val),
                        step=int(step_val),
                    ))
                else:
                    ranges.append(ParameterRange(
                        name=name,
                        min_value=min_val,
                        max_value=max_val,
                        step=step_val,
                    ))
    except Exception as e:
        print(f"解析失败: {line}, {e}")

print("\n解析结果:")
for r in ranges:
    print(f"  {r.name}: {r.min_value} ~ {r.max_value}, 步长={r.step}")

print("\n生成的参数值:")
for r in ranges:
    values = r.get_values()
    print(f"  {r.name}: {values[:5]}... (共{len(values)}个)")
