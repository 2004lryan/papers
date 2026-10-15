"""
55fetch_pmc_tables.py: 从开放获取(PMC)论文抓取【真实】in-article 数据表（温度×时间×品质）

诚实性：仅提取论文真实发表的 HTML 数据表（pandas.read_html），不构造任何数值。
逐表打印 shape/列/前几行，由人工判断哪张表含温度×时间×品质，再决定是否用于 C5 验证。
目标候选（已确认有可读数表/时间序列）：
  PMC9319022  长寿命食品 -18/5/25/40°C × 0-24月（Tables 1-4）
  PMC11941398 番茄 4/14/24°C × 0/1/5/9/15天（多为图，但查有无补充表）
  PMC7760833  樱桃番茄 基因型×温度×时间
  PMC10253207 苹果 2/4/6/8°C × 0-48h firmness（Appendix Tables A1/A2，已确认 in-article）

运行(服务器,需联网): python 55fetch_pmc_tables.py
输出: 打印各表概览 + 把疑似 温度×时间×品质 表存 CSV 到 ./pmc_tables/
"""
from __future__ import annotations
import io, sys, re, urllib.request
from pathlib import Path
import pandas as pd

OUTDIR = Path("./pmc_tables"); OUTDIR.mkdir(exist_ok=True)
PMCS = ["PMC10253207", "PMC9319022", "PMC11941398", "PMC7760833"]


def fetch(pmc):
    url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc}/"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research data extraction)"})
    return urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "ignore")


def main():
    for pmc in PMCS:
        print(f"\n{'='*60}\n{pmc}\n{'='*60}")
        try:
            html = fetch(pmc)
        except Exception as e:
            print(f"  fetch FAILED: {e}"); continue
        print(f"  html_len={len(html)} n_table_tags={html.count('<table')}")
        try:
            tabs = pd.read_html(io.StringIO(html))
        except Exception as e:
            print(f"  read_html FAILED: {e}"); continue
        print(f"  pandas parsed {len(tabs)} tables")
        for i, t in enumerate(tabs):
            cols = " | ".join(str(c)[:22] for c in list(t.columns)[:6])
            print(f"\n  --- T{i} shape={t.shape} ---\n  cols: {cols}")
            # 打印前3行用于人工判断
            with pd.option_context("display.max_columns", 8, "display.width", 160):
                print("  " + t.head(3).to_string().replace("\n", "\n  "))
            # 存所有表供后续精确解析
            try:
                t.to_csv(OUTDIR / f"{pmc}_T{i}.csv", index=False)
            except Exception:
                pass
    print(f"\n[saved all parsed tables to {OUTDIR.resolve()}]")


if __name__ == "__main__":
    main()
