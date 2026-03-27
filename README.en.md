## Travel-Agent

Weather- and map-augmented travel planner focused on college-friendly budgets. DeepSeek-compatible chat for reasoning, OpenWeatherMap for 5-day forecasts, and AMap for POI search and routing.

### Features
- Hour-by-hour itineraries as Markdown tables, including budget summaries and practical tips.
- Weather: prefetch up to 5 days (OWM 5-day/3-hour forecast) of precipitation probability and temperature; the model reuses these to avoid duplicate calls.
- POI search: AMap-based search by city and keywords, with price tiers (budget/mid/premium/any).
- Routing: AMap routes; if duration >120 minutes or missing, suggest nearer POIs or faster modes; walk/transit/drive can auto-fallback.
- Province/ambiguous cities: defaults to the provincial capital or a popular hub and states the choice.
- Debug logs: tool calls are emitted as JSON lines to stderr for easy grepping.

### How to Run
1) Setup (recommended: uv):
   ```bash
   uv sync
   ```
2) Environment variables:
   - `DEEPSEEK_API_KEY` (required)
   - `DEEPSEEK_BASE_URL` (optional, default https://api.deepseek.com)
   - `OPENWEATHERMAP_API_KEY` (required for weather)
   - `AMAP_API_KEY` (required for places and routing)
3) Examples:
   ```bash
   uv run main.py --city Shanghai --budget 3000 --days 3
   ```

### Current Limitations
- Weather limited to a 5-day window (including today); dates beyond that are unavailable.
- Dependent on AMap/OWM API availability; quota or network issues can leave gaps the model cannot fill.
- Routes over 120 minutes or missing only yield guidance to pick nearer POIs; no forced feasible path.
- Province/ambiguous city defaults may differ from user intent.
- Debug logs cover chat-time tool calls; weather prefetch happens before the chat loop and is not logged there.
