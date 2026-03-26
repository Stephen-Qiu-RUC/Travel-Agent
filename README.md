## Travel-Agent

Weather- and map-augmented travel planner for college-budget itineraries. Uses DeepSeek-compatible chat for reasoning, OpenWeatherMap for 5-day forecasts, and AMap for POI search and routing.

### 功能
- 小时级行程规划：根据城市、预算、旅行天数生成 Markdown 表格行程，附预算总结与实用贴士。
- 天气查询：预取未来最多 5 天（OWM 5-day/3-hour 预报）降水概率与温度，模型复用结果减少重复调用。
- POI 搜索：基于高德地图按城市与关键词检索景点/美食，可选择价格档位（budget/mid/premium/any）。
- 路线计算：高德路线，若耗时 >120 分钟或缺失则提示更换更近 POI 或更快方式；步行/公共交通/驾车自动 fallback。
- 省级/模糊城市处理：若输入为省级或不明确，模型默认使用省会或热门城市并在回答中声明。
- 调试输出：工具调用以 JSON 行写入 stderr（便于 grep）。

### 运行方法
1) 准备环境（推荐使用 uv）：
	```bash
	uv sync
	```
2) 设置环境变量：
	- `DEEPSEEK_API_KEY`（必需）
	- `DEEPSEEK_BASE_URL`（可选，默认 https://api.deepseek.com）
	- `OPENWEATHERMAP_API_KEY`（必需，用于天气）
	- `AMAP_API_KEY`（必需，用于地点与路线）
3) 运行示例：
	```bash
	uv run main.py --city 上海 --budget 3000 --days 3
	# 可选模型指定：
	uv run main.py --city 广州 --budget 2000 --days 2 --model deepseek-chat
	```

### 当前局限
- 天气仅支持未来 5 天窗口（含今天）；超出窗口不会返回更远日期。
- 依赖高德/OWM API 可用性，限额或网络异常会导致数据缺失，模型会提示但无法补全真实数据。
- 路线耗时 >120 分钟或未找到时只给出提示/改近建议，不会强行生成可行路径。
- 省级/模糊城市由模型选择省会/热门城市，可能与用户期望不完全一致。
- 工具调用调试日志仅覆盖对话阶段；预取天气在启动前完成，不会出现在聊天阶段日志中。
