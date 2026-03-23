#!/usr/bin/env python3

import json
from datetime import datetime
from pathlib import Path


ROOT = Path("/Users/yujeong/Desktop/AIOSS_GovOn")
HISTORY_PATH = ROOT / "docs/dora/history.json"
OUTPUT_PATH = ROOT / "docs/images/dora-dashboard.svg"


def format_number(value: float, suffix: str) -> str:
    return f"{value:.1f}{suffix}" if suffix != "/wk" else f"{value:.2f}{suffix}"


def main() -> None:
    history = json.loads(HISTORY_PATH.read_text())
    entries = history["entries"]
    latest = entries[-1]

    labels = [entry["date"][5:] for entry in entries]
    lead_points = [entry["lead_time_hours"] for entry in entries]
    deploy_points = [entry["deployment_frequency_per_week"] for entry in entries]
    mttr_points = [entry["mttr_hours"] for entry in entries]
    cfr_points = [entry["change_failure_rate"] for entry in entries]

    generated_at = datetime.fromisoformat(history["generated_at"].replace("Z", "+00:00"))
    generated_label = generated_at.strftime("%Y-%m-%d %H:%M UTC")

    def polyline(points, left, top, width, height):
      max_value = max(max(points), 1)
      step = width / (len(points) - 1 if len(points) > 1 else 1)
      coords = []
      for idx, value in enumerate(points):
          x = left + idx * step
          y = top + height - ((value / max_value) * height if max_value else 0)
          coords.append(f"{x:.1f},{y:.1f}")
      return " ".join(coords)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="920" viewBox="0 0 1400 920" fill="none">
  <rect width="1400" height="920" fill="#F4EFE7"/>
  <circle cx="154" cy="130" r="190" fill="#115E59" fill-opacity="0.14"/>
  <circle cx="1228" cy="98" r="170" fill="#B45309" fill-opacity="0.16"/>
  <rect x="48" y="42" width="820" height="214" rx="28" fill="#FFFCF7" fill-opacity="0.94" stroke="#D9CFC1"/>
  <rect x="892" y="42" width="460" height="214" rx="28" fill="#173B3B"/>
  <text x="86" y="88" fill="#115E59" font-family="Arial, sans-serif" font-size="20" font-weight="700">Chart.js Dashboard</text>
  <text x="86" y="146" fill="#1D2A2A" font-family="Arial, sans-serif" font-size="54" font-weight="800">AIOSS_GovOn</text>
  <text x="86" y="202" fill="#1D2A2A" font-family="Arial, sans-serif" font-size="54" font-weight="800">DORA Metrics Dashboard</text>
  <text x="86" y="234" fill="#5E6A68" font-family="Arial, sans-serif" font-size="22">GitHub Actions 자동 수집 결과를 기반으로 생성한 README 미리보기 이미지</text>
  <text x="930" y="92" fill="#D5F3EE" font-family="Arial, sans-serif" font-size="20">Current Grade</text>
  <text x="930" y="188" fill="#FFFFFF" font-family="Arial, sans-serif" font-size="86" font-weight="800">{latest["grade"]}</text>
  <text x="930" y="228" fill="#CFE7E3" font-family="Arial, sans-serif" font-size="22">Primary branch: {latest["primary_branch"]} · Last {latest["window_days"]} days</text>

  <g font-family="Arial, sans-serif">
    <rect x="48" y="280" width="318" height="144" rx="24" fill="#FFFCF7" fill-opacity="0.94" stroke="#D9CFC1"/>
    <text x="76" y="324" fill="#5E6A68" font-size="22">Lead Time</text>
    <text x="76" y="386" fill="#1D2A2A" font-size="56" font-weight="800">{format_number(latest["lead_time_hours"], "h")}</text>
    <text x="76" y="408" fill="#5E6A68" font-size="18">PR first commit → merge average</text>

    <rect x="382" y="280" width="318" height="144" rx="24" fill="#FFFCF7" fill-opacity="0.94" stroke="#D9CFC1"/>
    <text x="410" y="324" fill="#5E6A68" font-size="22">Deployment Frequency</text>
    <text x="410" y="386" fill="#1D2A2A" font-size="56" font-weight="800">{format_number(latest["deployment_frequency_per_week"], "/wk")}</text>
    <text x="410" y="408" fill="#5E6A68" font-size="18">Window total deploys: {latest["deployment_frequency_window_total"]}</text>

    <rect x="716" y="280" width="318" height="144" rx="24" fill="#FFFCF7" fill-opacity="0.94" stroke="#D9CFC1"/>
    <text x="744" y="324" fill="#5E6A68" font-size="22">MTTR</text>
    <text x="744" y="386" fill="#1D2A2A" font-size="56" font-weight="800">{format_number(latest["mttr_hours"], "h")}</text>
    <text x="744" y="408" fill="#5E6A68" font-size="18">bug issue open → close average</text>

    <rect x="1050" y="280" width="302" height="144" rx="24" fill="#FFFCF7" fill-opacity="0.94" stroke="#D9CFC1"/>
    <text x="1078" y="324" fill="#5E6A68" font-size="22">Change Failure Rate</text>
    <text x="1078" y="386" fill="#1D2A2A" font-size="56" font-weight="800">{format_number(latest["change_failure_rate"], "%")}</text>
    <text x="1078" y="408" fill="#5E6A68" font-size="18">hotfix/revert commit ratio</text>
  </g>

  <g font-family="Arial, sans-serif">
    <rect x="48" y="448" width="644" height="192" rx="24" fill="#FFFCF7" fill-opacity="0.94" stroke="#D9CFC1"/>
    <text x="76" y="486" fill="#1D2A2A" font-size="26" font-weight="700">Lead Time Trend</text>
    <polyline points="{polyline(lead_points, 102, 520, 540, 74)}" stroke="#115E59" stroke-width="8" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    <line x1="102" y1="594" x2="642" y2="594" stroke="#D7D0C4" stroke-width="2"/>
    <text x="102" y="620" fill="#5E6A68" font-size="18">{labels[0]}</text>
    <text x="358" y="620" fill="#5E6A68" font-size="18">{labels[1]}</text>
    <text x="602" y="620" fill="#5E6A68" font-size="18">{labels[2]}</text>

    <rect x="708" y="448" width="644" height="192" rx="24" fill="#FFFCF7" fill-opacity="0.94" stroke="#D9CFC1"/>
    <text x="736" y="486" fill="#1D2A2A" font-size="26" font-weight="700">Deployment Frequency Trend</text>
    <polyline points="{polyline(deploy_points, 762, 520, 534, 74)}" stroke="#B45309" stroke-width="8" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    <line x1="762" y1="594" x2="1296" y2="594" stroke="#D7D0C4" stroke-width="2"/>
    <text x="780" y="620" fill="#5E6A68" font-size="18">{labels[0]}</text>
    <text x="958" y="620" fill="#5E6A68" font-size="18">{labels[1]}</text>
    <text x="1136" y="620" fill="#5E6A68" font-size="18">{labels[2]}</text>

    <rect x="48" y="664" width="644" height="192" rx="24" fill="#FFFCF7" fill-opacity="0.94" stroke="#D9CFC1"/>
    <text x="76" y="702" fill="#1D2A2A" font-size="26" font-weight="700">MTTR Trend</text>
    <polyline points="{polyline(mttr_points, 102, 736, 540, 74)}" stroke="#1D4ED8" stroke-width="8" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    <line x1="102" y1="810" x2="642" y2="810" stroke="#D7D0C4" stroke-width="2"/>
    <text x="102" y="836" fill="#5E6A68" font-size="18">{labels[0]}</text>
    <text x="358" y="836" fill="#5E6A68" font-size="18">{labels[1]}</text>
    <text x="602" y="836" fill="#5E6A68" font-size="18">{labels[2]}</text>

    <rect x="708" y="664" width="644" height="192" rx="24" fill="#FFFCF7" fill-opacity="0.94" stroke="#D9CFC1"/>
    <text x="736" y="702" fill="#1D2A2A" font-size="26" font-weight="700">Change Failure Rate Trend</text>
    <polyline points="{polyline(cfr_points, 762, 736, 534, 74)}" stroke="#B91C1C" stroke-width="8" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    <line x1="762" y1="810" x2="1296" y2="810" stroke="#D7D0C4" stroke-width="2"/>
    <text x="780" y="836" fill="#5E6A68" font-size="18">{labels[0]}</text>
    <text x="958" y="836" fill="#5E6A68" font-size="18">{labels[1]}</text>
    <text x="1136" y="836" fill="#5E6A68" font-size="18">{labels[2]}</text>
  </g>

  <text x="48" y="892" fill="#5E6A68" font-family="Arial, sans-serif" font-size="18">Generated from docs/dora/history.json · {generated_label}</text>
</svg>
"""

    OUTPUT_PATH.write_text(svg)


if __name__ == "__main__":
    main()
